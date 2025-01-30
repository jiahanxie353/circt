//===- MemoryBanking.cpp - memory bank parallel loops -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements parallel loop memory banking.
//
//===----------------------------------------------------------------------===//

#include "circt/Support/LLVM.h"
#include "circt/Transforms/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FormatVariadic.h"
#include <numeric>

namespace circt {
#define GEN_PASS_DEF_MEMORYBANKING
#include "circt/Transforms/Passes.h.inc"
} // namespace circt

using namespace mlir;
using namespace circt;

namespace {

/// Partition memories used in `affine.parallel` operation by the
/// `bankingFactor` throughout the program.
struct MemoryBankingPass
    : public circt::impl::MemoryBankingBase<MemoryBankingPass> {
  MemoryBankingPass(const MemoryBankingPass &other) = default;
  explicit MemoryBankingPass(ArrayRef<unsigned> bankingFactors = {},
                             ArrayRef<unsigned> bankingDimensions = {}) {}

  void runOnOperation() override;

private:
  // map from original memory definition to newly allocated banks
  DenseMap<Value, DenseMap<unsigned, SmallVector<Value>>> memoryToBanks;
  DenseSet<Operation *> opsToErase;
  // Track memory references that need to be cleaned up after memory banking is
  // complete.
  DenseSet<Value> oldMemRefVals;
};
} // namespace

// Collect all memref in the `parOp`'s region'
DenseSet<Value> collectMemRefs(mlir::affine::AffineParallelOp parOp) {
  DenseSet<Value> memrefVals;
  parOp.walk([&](Operation *op) {
    for (auto operand : op->getOperands()) {
      if (isa<MemRefType>(operand.getType()))
        memrefVals.insert(operand);
    }
    return WalkResult::advance();
  });
  return memrefVals;
}

// Verify the banking configuration with different conditions.
void verifyBankingConfigurations(ArrayRef<unsigned> bankingDimensions,
                                 ArrayRef<unsigned> bankingFactors,
                                 MemRefType originalType) {
  ArrayRef<int64_t> originalShape = originalType.getShape();
  assert(!originalShape.empty() && "MemRef shape should not be empty");

  assert(bankingDimensions.size() == bankingFactors.size() &&
         ("Banking dimensions and factors must be of the same size. Provided "
          "dimensions: " +
          Twine(bankingDimensions.size()) +
          ", factors: " + Twine(bankingFactors.size()))
             .str()
             .c_str());

  for (size_t i = 0; i < bankingDimensions.size(); ++i) {
    assert(bankingDimensions[i] < originalType.getRank() &&
           ("Banking dimension " + Twine(bankingDimensions[i]) +
            " is out of bounds for MemRef rank " +
            Twine(originalType.getRank()))
               .str()
               .c_str());

    assert(originalShape[bankingDimensions[i]] % bankingFactors[i] == 0 &&
           ("MemRef shape dimension " +
            Twine(originalShape[bankingDimensions[i]]) + " at index " +
            Twine(bankingDimensions[i]) +
            " is not evenly divisible by the banking factor " +
            Twine(bankingFactors[i]))
               .str()
               .c_str());
  }
}

MemRefType computeBankedMemRefType(MemRefType originalType,
                                   ArrayRef<unsigned> bankingFactors,
                                   ArrayRef<unsigned> bankingDimensions) {
  ArrayRef<int64_t> originalShape = originalType.getShape();
  SmallVector<int64_t, 4> newShape(originalShape.begin(), originalShape.end());

  for (size_t i = 0; i < bankingDimensions.size(); ++i)
    newShape[bankingDimensions[i]] /= bankingFactors[i];

  return MemRefType::get(newShape, originalType.getElementType(),
                         originalType.getLayout(),
                         originalType.getMemorySpace());
}

// Decodes the flat index `linIndex` into an n-dimensional index based on the
// given `shape` of the array in row-major order. Returns an array to represent
// the n-dimensional indices.
SmallVector<int64_t> decodeIndex(int64_t linIndex, ArrayRef<int64_t> shape) {
  const unsigned rank = shape.size();
  SmallVector<int64_t> ndIndex(rank, 0);

  // Compute from last dimension to first because we assume row-major.
  for (int64_t d = rank - 1; d >= 0; --d) {
    ndIndex[d] = linIndex % shape[d];
    linIndex /= shape[d];
  }

  return ndIndex;
}

// Performs multi-dimensional slicing on `allAttrs` by extracting all elements
// whose coordinates range from `bankCnt`*`bankingDimension` to
// (`bankCnt`+1)*`bankingDimension` from `bankingDimension`'s dimension, leaving
// other dimensions alone.
SmallVector<SmallVector<Attribute>>
sliceSubBlock(ArrayRef<Attribute> allAttrs, ArrayRef<int64_t> memShape,
              ArrayRef<unsigned> bankingDimensions,
              ArrayRef<unsigned> bankingFactors) {
  size_t numElements = std::reduce(memShape.begin(), memShape.end(), 1,
                                   std::multiplies<size_t>());

  // Create a vector of sub-blocks to store banked attributes
  SmallVector<SmallVector<Attribute>> subBlocks;
  size_t numBanks =
      std::accumulate(bankingFactors.begin(), bankingFactors.end(), 1,
                      std::multiplies<size_t>());
  subBlocks.resize(numBanks);

  for (unsigned linIndex = 0; linIndex < numElements; ++linIndex) {
    SmallVector<int64_t> ndIndex = decodeIndex(linIndex, memShape);

    unsigned subBlockIndex = 0;
    unsigned stride = 1;
    for (size_t i = 0; i < bankingDimensions.size(); ++i) {
      unsigned dimIdx = bankingDimensions[i];
      subBlockIndex += (ndIndex[dimIdx] % bankingFactors[i]) * stride;
      stride *= bankingFactors[i];
    }

    subBlocks[subBlockIndex].push_back(allAttrs[linIndex]);
  }

  return subBlocks;
}

// Handles the splitting of a GetGlobalOp into multiple banked memory and
// creates new GetGlobalOp to represent each banked memory by slicing the data
// in the original GetGlobalOp.
SmallVector<Value> handleGetGlobalOp(memref::GetGlobalOp getGlobalOp,
                                     ArrayRef<unsigned> bankingFactors,
                                     ArrayRef<unsigned> bankingDimensions,
                                     MemRefType newMemRefType,
                                     OpBuilder &builder) {
  SmallVector<Value> banks;
  auto memTy = cast<MemRefType>(getGlobalOp.getType());
  ArrayRef<int64_t> originalShape = memTy.getShape();

  SmallVector<int64_t> newShape(originalShape.begin(), originalShape.end());
  for (size_t i = 0; i < bankingDimensions.size(); ++i)
    newShape[bankingDimensions[i]] /= bankingFactors[i];

  size_t totalBanks =
      std::accumulate(bankingFactors.begin(), bankingFactors.end(), 1,
                      std::multiplies<size_t>());

  auto *symbolTableOp = getGlobalOp->getParentWithTrait<OpTrait::SymbolTable>();
  auto globalOpNameAttr = getGlobalOp.getNameAttr();
  auto globalOp = dyn_cast_or_null<memref::GlobalOp>(
      SymbolTable::lookupSymbolIn(symbolTableOp, globalOpNameAttr));
  assert(globalOp && "The corresponding GlobalOp should exist in the module");

  auto cstAttr =
      dyn_cast_or_null<DenseElementsAttr>(globalOp.getConstantInitValue());
  auto attributes = cstAttr.getValues<Attribute>();
  SmallVector<Attribute, 8> allAttrs(attributes.begin(), attributes.end());

  auto subBlocks =
      sliceSubBlock(allAttrs, originalShape, bankingDimensions, bankingFactors);

  builder.setInsertionPointAfter(globalOp);
  OpBuilder::InsertPoint globalOpsInsertPt = builder.saveInsertionPoint();
  builder.setInsertionPointAfter(getGlobalOp);
  OpBuilder::InsertPoint getGlobalOpsInsertPt = builder.saveInsertionPoint();

  for (size_t bankCnt = 0; bankCnt < totalBanks; ++bankCnt) {
    std::string newName =
        llvm::formatv("{0}_bank_{1}", globalOpNameAttr.getValue(), bankCnt);

    auto tensorType =
        RankedTensorType::get({newShape}, globalOp.getType().getElementType());
    auto newInitValue = DenseElementsAttr::get(tensorType, subBlocks[bankCnt]);

    builder.restoreInsertionPoint(globalOpsInsertPt);
    auto newGlobalOp = builder.create<memref::GlobalOp>(
        globalOp.getLoc(), builder.getStringAttr(newName),
        globalOp.getSymVisibilityAttr(), TypeAttr::get(newMemRefType),
        newInitValue, globalOp.getConstantAttr(), globalOp.getAlignmentAttr());
    builder.setInsertionPointAfter(newGlobalOp);
    globalOpsInsertPt = builder.saveInsertionPoint();

    builder.restoreInsertionPoint(getGlobalOpsInsertPt);
    auto newGetGlobalOp = builder.create<memref::GetGlobalOp>(
        getGlobalOp.getLoc(), newMemRefType, newGlobalOp.getName());
    builder.setInsertionPointAfter(newGetGlobalOp);
    getGlobalOpsInsertPt = builder.saveInsertionPoint();

    banks.push_back(newGetGlobalOp);
  }

  globalOp.erase();
  return banks;
}

SmallVector<unsigned>
getSpecifiedOrDefaultBankingDims(ArrayRef<unsigned> bankingDimensions,
                                 int64_t rank, ArrayRef<int64_t> shape) {
  // If the banking dimensions are already specified, return it.
  if (!bankingDimensions.empty())
    return SmallVector<unsigned>{bankingDimensions.begin(),
                                 bankingDimensions.end()};

  // Otherwise, find the innermost dimension with size > 1.
  // For example, [[1], [2], [3], [4]] with `bankingFactor`=2 will be banked to
  // [[1], [3]] and [[2], [4]].
  int bankingDimension = -1;
  for (int dim = rank - 1; dim >= 0; --dim) {
    if (shape[dim] > 1) {
      bankingDimension = dim;
      break;
    }
  }

  assert(bankingDimension >= 0 && "No eligible dimension for banking");
  return {static_cast<unsigned>(bankingDimension)};
}

// Retrieve potentially specified banking factor/dimension attributes and
// _always_ overwrite the command line or the default ones.
void resolveBankingAttributes(Value originalMem,
                              MutableArrayRef<unsigned> bankingFactors,
                              MutableArrayRef<unsigned> bankingDimensions) {
  auto parseBankingAttr = [](Operation *op, StringRef attrName,
                             MutableArrayRef<unsigned> &result) {
    if (auto attr = dyn_cast_if_present<ArrayAttr>(op->getAttr(attrName))) {
      for (size_t i = 0; i < attr.size(); ++i) {
        result[i] = cast<IntegerAttr>(attr[i]).getInt();
      }
    } else if (auto intAttr =
                   dyn_cast_if_present<IntegerAttr>(op->getAttr(attrName))) {
      result[0] = intAttr.getInt();
    }
  };

  if (auto *originalDef = originalMem.getDefiningOp()) {
    parseBankingAttr(originalDef, "banking.factor", bankingFactors);
    parseBankingAttr(originalDef, "banking.dimension", bankingDimensions);
    return;
  }

  if (auto blockArg = dyn_cast<BlockArgument>(originalMem)) {
    auto *parentOp = blockArg.getOwner()->getParentOp();
    auto funcOp = dyn_cast<func::FuncOp>(parentOp);
    assert(funcOp &&
           "Expected the original memory to be a FuncOp block argument!");

    unsigned argIndex = blockArg.getArgNumber();
    if (auto argAttrs = funcOp.getArgAttrDict(argIndex)) {
      parseBankingAttr(funcOp, "banking.factor", bankingFactors);
      parseBankingAttr(funcOp, "banking.dimension", bankingDimensions);
    }
  }
}

// Update the argument types of `funcOp` by inserting `numInsertedArgs` number
// of `newMemRefType` after `argIndex`.
void updateFuncOpArgumentTypes(func::FuncOp funcOp, unsigned argIndex,
                               MemRefType newMemRefType,
                               unsigned numInsertedArgs) {
  auto originalArgTypes = funcOp.getFunctionType().getInputs();
  SmallVector<Type, 4> updatedArgTypes;

  // Rebuild the argument types, inserting new types for the newly added
  // arguments
  for (unsigned i = 0; i < originalArgTypes.size(); ++i) {
    updatedArgTypes.push_back(originalArgTypes[i]);

    // Insert new argument types after the specified argument index
    if (i == argIndex) {
      for (unsigned j = 0; j < numInsertedArgs; ++j) {
        updatedArgTypes.push_back(newMemRefType);
      }
    }
  }

  // Update the function type with the new argument types
  auto resultTypes = funcOp.getFunctionType().getResults();
  auto newFuncType =
      FunctionType::get(funcOp.getContext(), updatedArgTypes, resultTypes);
  funcOp.setType(newFuncType);
}

// Update `funcOp`'s "arg_attrs" by inserting `numInsertedArgs` number of empty
// DictionaryAttr after `argIndex`.
void updateFuncOpArgAttrs(func::FuncOp funcOp, unsigned argIndex,
                          unsigned numInsertedArgs) {
  ArrayAttr existingArgAttrs = funcOp->getAttrOfType<ArrayAttr>("arg_attrs");
  SmallVector<Attribute, 4> updatedArgAttrs;
  unsigned numArguments = funcOp.getNumArguments();
  unsigned newNumArguments = numArguments + numInsertedArgs;
  updatedArgAttrs.resize(newNumArguments);

  // Copy existing attributes, adjusting for the new arguments
  for (unsigned i = 0; i < numArguments; ++i) {
    // Shift attributes for arguments after the inserted ones.
    unsigned newIndex = (i > argIndex) ? i + numInsertedArgs : i;
    updatedArgAttrs[newIndex] = existingArgAttrs
                                    ? existingArgAttrs[i]
                                    : DictionaryAttr::get(funcOp.getContext());
  }

  // Initialize new attributes for the inserted arguments as empty dictionaries
  for (unsigned i = 0; i < numInsertedArgs; ++i) {
    updatedArgAttrs[argIndex + 1 + i] =
        DictionaryAttr::get(funcOp.getContext());
  }

  // Set the updated attributes.
  funcOp->setAttr("arg_attrs",
                  ArrayAttr::get(funcOp.getContext(), updatedArgAttrs));
}

DenseMap<unsigned, SmallVector<Value>>
createBanks(Value originalMem, MutableArrayRef<unsigned> bankingFactors,
            MutableArrayRef<unsigned> bankingDimensions) {
  MemRefType originalMemRefType = cast<MemRefType>(originalMem.getType());
  unsigned rank = originalMemRefType.getRank();
  ArrayRef<int64_t> shape = originalMemRefType.getShape();

  SmallVector<unsigned> updatedBankingDims =
      getSpecifiedOrDefaultBankingDims(bankingDimensions, rank, shape);

  resolveBankingAttributes(originalMem, bankingFactors, updatedBankingDims);

  verifyBankingConfigurations(updatedBankingDims, bankingFactors,
                              originalMemRefType);

  // A map to store banks for each banking dimension
  DenseMap<unsigned, SmallVector<Value>> banksPerDim;

  MemRefType newMemRefType = computeBankedMemRefType(
      originalMemRefType, bankingFactors, updatedBankingDims);

  if (auto blockArgMem = dyn_cast<BlockArgument>(originalMem)) {
    Block *block = blockArgMem.getOwner();
    unsigned blockArgNum = blockArgMem.getArgNumber();

    for (size_t dimIdx = 0; dimIdx < updatedBankingDims.size(); ++dimIdx) {
      unsigned bankingDim = updatedBankingDims[dimIdx];
      unsigned numBanks = bankingFactors[dimIdx];

      SmallVector<Value> banks;
      for (unsigned i = 0; i < numBanks; ++i)
        block->insertArgument(blockArgNum + 1 + i, newMemRefType,
                              blockArgMem.getLoc());

      auto blockArgs = block->getArguments().slice(blockArgNum + 1, numBanks);
      banks.append(blockArgs.begin(), blockArgs.end());

      banksPerDim[bankingDim] = banks;
    }

    auto *parentOp = block->getParentOp();
    auto funcOp = dyn_cast<func::FuncOp>(parentOp);
    assert(funcOp && "BlockArgument is not part of a FuncOp");

    // Update function argument types & attributes
    updateFuncOpArgumentTypes(funcOp, blockArgNum, newMemRefType,
                              bankingFactors[0]);
    updateFuncOpArgAttrs(funcOp, blockArgNum, bankingFactors[0]);

  } else {
    Operation *originalDef = originalMem.getDefiningOp();
    Location loc = originalDef->getLoc();
    OpBuilder builder(originalDef);
    builder.setInsertionPointAfter(originalDef);

    TypeSwitch<Operation *>(originalDef)
        .Case<memref::AllocOp>([&](memref::AllocOp allocOp) {
          for (size_t dimIdx = 0; dimIdx < updatedBankingDims.size();
               ++dimIdx) {
            unsigned bankingDim = updatedBankingDims[dimIdx];
            unsigned numBanks = bankingFactors[dimIdx];

            SmallVector<Value> banks;
            for (uint64_t bankCnt = 0; bankCnt < numBanks; ++bankCnt) {
              auto bankAllocOp =
                  builder.create<memref::AllocOp>(loc, newMemRefType);
              banks.push_back(bankAllocOp);
            }
            banksPerDim[bankingDim] = banks;
          }
        })
        .Case<memref::AllocaOp>([&](memref::AllocaOp allocaOp) {
          for (size_t dimIdx = 0; dimIdx < updatedBankingDims.size();
               ++dimIdx) {
            unsigned bankingDim = updatedBankingDims[dimIdx];
            unsigned numBanks = bankingFactors[dimIdx];

            SmallVector<Value> banks;
            for (uint64_t bankCnt = 0; bankCnt < numBanks; ++bankCnt) {
              auto bankAllocaOp =
                  builder.create<memref::AllocaOp>(loc, newMemRefType);
              banks.push_back(bankAllocaOp);
            }
            banksPerDim[bankingDim] = banks;
          }
        })
        .Case<memref::GetGlobalOp>([&](memref::GetGlobalOp getGlobalOp) {
          for (size_t dimIdx = 0; dimIdx < updatedBankingDims.size();
               ++dimIdx) {
            unsigned bankingDim = updatedBankingDims[dimIdx];
            unsigned numBanks = bankingFactors[dimIdx];

            auto newBanks = handleGetGlobalOp(getGlobalOp, numBanks, bankingDim,
                                              newMemRefType, builder);
            banksPerDim[bankingDim] = newBanks;
          }
        })
        .Default([](Operation *) {
          llvm_unreachable("Unhandled memory operation type");
        });
  }

  return banksPerDim;
}

// Replace the original load operations with newly created memory banks
struct BankAffineLoadPattern
    : public OpRewritePattern<mlir::affine::AffineLoadOp> {
  BankAffineLoadPattern(
      MLIRContext *context, MutableArrayRef<unsigned> bankingFactors,
      MutableArrayRef<unsigned> bankingDimensions,
      DenseMap<Value, DenseMap<unsigned, SmallVector<Value>>> &memoryToBanks,
      DenseSet<Value> &oldMemRefVals)
      : OpRewritePattern<mlir::affine::AffineLoadOp>(context),
        bankingFactors(bankingFactors), bankingDimensions(bankingDimensions),
        memoryToBanks(memoryToBanks), oldMemRefVals(oldMemRefVals) {}

  LogicalResult matchAndRewrite(mlir::affine::AffineLoadOp loadOp,
                                PatternRewriter &rewriter) const override {
    Location loc = loadOp.getLoc();
    auto originalMem = loadOp.getMemref();

    if (!memoryToBanks.count(originalMem))
      return failure();
    auto &banksPerDim = memoryToBanks[originalMem];

    auto loadIndices = loadOp.getIndices();
    MemRefType originalMemRefType = loadOp.getMemRefType();
    int64_t memrefRank = originalMemRefType.getRank();
    ArrayRef<int64_t> shape = originalMemRefType.getShape();

    SmallVector<unsigned> updatedBankingDims =
        getSpecifiedOrDefaultBankingDims(bankingDimensions, memrefRank, shape);

    resolveBankingAttributes(originalMem, bankingFactors, updatedBankingDims);

    verifyBankingConfigurations(updatedBankingDims, bankingFactors,
                                originalMemRefType);

    // Compute new indices and bank selection logic for all banking dimensions
    SmallVector<Value> selectedBanks;
    SmallVector<Value, 4> newIndices(loadIndices.begin(), loadIndices.end());

    for (size_t dimIdx = 0; dimIdx < updatedBankingDims.size(); ++dimIdx) {
      unsigned bankingDim = updatedBankingDims[dimIdx];
      unsigned numBanks = bankingFactors[dimIdx];

      auto modMap =
          AffineMap::get(/*dimCount=*/memrefRank, /*symbolCount=*/0,
                         {rewriter.getAffineDimExpr(bankingDim) % numBanks});
      auto divMap = AffineMap::get(
          memrefRank, 0,
          {rewriter.getAffineDimExpr(bankingDim).floorDiv(numBanks)});

      Value bankIndex =
          rewriter.create<affine::AffineApplyOp>(loc, modMap, loadIndices);
      Value offset =
          rewriter.create<affine::AffineApplyOp>(loc, divMap, loadIndices);
      newIndices[bankingDim] = offset;
      // Use `bankIndex` to select the correct bank.
      selectedBanks.push_back(bankIndex);
    }

    Value selectedMemRef = originalMem;
    for (size_t dimIdx = 0; dimIdx < updatedBankingDims.size(); ++dimIdx) {
      unsigned bankingDim = updatedBankingDims[dimIdx];
      auto &banks = banksPerDim[bankingDim];

      SmallVector<Type> resultTypes = {selectedMemRef.getType()};
      SmallVector<int64_t> caseValues =
          llvm::to_vector(llvm::seq<int64_t>(0, bankingFactors[dimIdx]));

      // Create switch operation to select the correct bank.
      rewriter.setInsertionPoint(loadOp);
      scf::IndexSwitchOp switchOp = rewriter.create<scf::IndexSwitchOp>(
          loc, resultTypes, selectedBanks[dimIdx], caseValues,
          bankingFactors[dimIdx]);
      for (unsigned i = 0; i < bankingFactors[dimIdx]; ++i) {
        Region &caseRegion = switchOp.getCaseRegions()[i];
        rewriter.setInsertionPointToStart(&caseRegion.emplaceBlock());
        rewriter.create<scf::YieldOp>(loc, banks[i]);
      }

      Region &defaultRegion = switchOp.getDefaultRegion();
      assert(defaultRegion.empty() && "Default region should be empty");
      rewriter.setInsertionPointToStart(&defaultRegion.emplaceBlock());

      TypedAttr zeroAttr =
          cast<TypedAttr>(rewriter.getZeroAttr(loadOp.getType()));
      auto defaultValue = rewriter.create<arith::ConstantOp>(loc, zeroAttr);
      rewriter.create<scf::YieldOp>(loc, defaultValue.getResult());

      // We track Load's memory reference only if it is a block argument - this
      // is the only case where the reference isn't replaced.
      if (Value memRef = loadOp.getMemref(); isa<BlockArgument>(memRef))
        oldMemRefVals.insert(memRef);

      selectedMemRef = switchOp.getResult(0);

      // Perform load operation on the selected memory bank.
      Value bankedLoad = rewriter.create<mlir::affine::AffineLoadOp>(
          loc, selectedMemRef, newIndices);
      rewriter.replaceOp(loadOp, bankedLoad);
    }

    return success();
  }

private:
  MutableArrayRef<unsigned> bankingFactors;
  MutableArrayRef<unsigned> bankingDimensions;
  DenseMap<Value, DenseMap<unsigned, SmallVector<Value>>> &memoryToBanks;
  DenseSet<Value> &oldMemRefVals;
};

// Replace the original store operations with newly created memory banks
struct BankAffineStorePattern
    : public OpRewritePattern<mlir::affine::AffineStoreOp> {
  BankAffineStorePattern(
      MLIRContext *context, MutableArrayRef<unsigned> bankingFactors,
      MutableArrayRef<unsigned> bankingDimensions,
      DenseMap<Value, DenseMap<unsigned, SmallVector<Value>>> &memoryToBanks,
      DenseSet<Operation *> &opsToErase, DenseSet<Operation *> &processedOps,
      DenseSet<Value> &oldMemRefVals)
      : OpRewritePattern<mlir::affine::AffineStoreOp>(context),
        bankingFactors(bankingFactors), bankingDimensions(bankingDimensions),
        memoryToBanks(memoryToBanks), opsToErase(opsToErase),
        processedOps(processedOps), oldMemRefVals(oldMemRefVals) {}

  LogicalResult matchAndRewrite(mlir::affine::AffineStoreOp storeOp,
                                PatternRewriter &rewriter) const override {
    if (processedOps.contains(storeOp)) {
      return failure();
    }
    Location loc = storeOp.getLoc();
    auto originalMem = storeOp.getMemref();
    if (!memoryToBanks.count(originalMem))
      return failure();
    auto &banksPerDim = memoryToBanks[originalMem];

    auto storeIndices = storeOp.getIndices();
    auto originalMemRefType = storeOp.getMemRefType();
    int64_t memrefRank = originalMemRefType.getRank();
    ArrayRef<int64_t> shape = originalMemRefType.getShape();

    SmallVector<unsigned> updatedBankingDims =
        getSpecifiedOrDefaultBankingDims(bankingDimensions, memrefRank, shape);

    resolveBankingAttributes(originalMem, bankingFactors, updatedBankingDims);

    verifyBankingConfigurations(updatedBankingDims, bankingFactors,
                                originalMemRefType);

    // Compute new indices and bank selection logic for all banking dimensions.
    SmallVector<Value> selectedBanks;
    SmallVector<Value, 4> newIndices(storeIndices.begin(), storeIndices.end());

    for (size_t dimIdx = 0; dimIdx < updatedBankingDims.size(); ++dimIdx) {
      unsigned bankingDim = updatedBankingDims[dimIdx];
      unsigned numBanks = bankingFactors[dimIdx];
      auto modMap = AffineMap::get(
          /*dimCount=*/memrefRank, /*symbolCount=*/0,
          {rewriter.getAffineDimExpr(bankingDim) % numBanks});
      auto divMap = AffineMap::get(
          memrefRank, 0,
          {rewriter.getAffineDimExpr(bankingDim).floorDiv(numBanks)});

      Value bankIndex =
          rewriter.create<affine::AffineApplyOp>(loc, modMap, storeIndices);
      Value offset =
          rewriter.create<affine::AffineApplyOp>(loc, divMap, storeIndices);
      newIndices[bankingDim] = offset;

      selectedBanks.push_back(bankIndex);
    }

    Value selectedMemRef = originalMem;
    for (size_t dimIdx = 0; dimIdx < updatedBankingDims.size(); ++dimIdx) {
      unsigned bankingDim = updatedBankingDims[dimIdx];
      auto &banks = banksPerDim[bankingDim];

      SmallVector<Type> resultTypes = {selectedMemRef.getType()};
      SmallVector<int64_t> caseValues =
          llvm::to_vector(llvm::seq<int64_t>(0, bankingFactors[dimIdx]));

      rewriter.setInsertionPoint(storeOp);
      scf::IndexSwitchOp switchOp = rewriter.create<scf::IndexSwitchOp>(
          loc, resultTypes, selectedBanks[dimIdx], caseValues,
          /*numRegions=*/bankingFactors[dimIdx]);

      for (unsigned i = 0; i < bankingFactors[dimIdx]; ++i) {
        Region &caseRegion = switchOp.getCaseRegions()[i];
        rewriter.setInsertionPointToStart(&caseRegion.emplaceBlock());
        rewriter.create<scf::YieldOp>(loc, banks[i]);
      }

      selectedMemRef = switchOp.getResult(0);

      rewriter.create<mlir::affine::AffineStoreOp>(
          loc, storeOp.getValueToStore(), selectedMemRef, newIndices);

      Region &defaultRegion = switchOp.getDefaultRegion();
      assert(defaultRegion.empty() && "Default region should be empty");
      rewriter.setInsertionPointToStart(&defaultRegion.emplaceBlock());

      rewriter.create<scf::YieldOp>(loc);
    }

    processedOps.insert(storeOp);
    opsToErase.insert(storeOp);
    oldMemRefVals.insert(storeOp.getMemref());

    return success();
  }

private:
  MutableArrayRef<unsigned> bankingFactors;
  MutableArrayRef<unsigned> bankingDimensions;
  DenseMap<Value, DenseMap<unsigned, SmallVector<Value>>> &memoryToBanks;
  DenseSet<Operation *> &opsToErase;
  DenseSet<Operation *> &processedOps;
  DenseSet<Value> &oldMemRefVals;
};

// Replace the original return operation with newly created memory banks
struct BankReturnPattern : public OpRewritePattern<func::ReturnOp> {
  BankReturnPattern(MLIRContext *context,
                    DenseMap<Value, SmallVector<Value>> &memoryToBanks)
      : OpRewritePattern<func::ReturnOp>(context),
        memoryToBanks(memoryToBanks) {}

  LogicalResult matchAndRewrite(func::ReturnOp returnOp,
                                PatternRewriter &rewriter) const override {
    Location loc = returnOp.getLoc();
    SmallVector<Value, 4> newReturnOperands;
    bool allOrigMemsUsedByReturn = true;
    for (auto operand : returnOp.getOperands()) {
      if (!memoryToBanks.contains(operand)) {
        newReturnOperands.push_back(operand);
        continue;
      }
      if (operand.hasOneUse())
        allOrigMemsUsedByReturn = false;
      auto banks = memoryToBanks[operand];
      newReturnOperands.append(banks.begin(), banks.end());
    }

    func::FuncOp funcOp = returnOp.getParentOp();
    rewriter.setInsertionPointToEnd(&funcOp.getBlocks().front());
    auto newReturnOp =
        rewriter.create<func::ReturnOp>(loc, ValueRange(newReturnOperands));
    TypeRange newReturnType = TypeRange(newReturnOperands);
    FunctionType newFuncType =
        FunctionType::get(funcOp.getContext(),
                          funcOp.getFunctionType().getInputs(), newReturnType);
    funcOp.setType(newFuncType);

    if (allOrigMemsUsedByReturn)
      rewriter.replaceOp(returnOp, newReturnOp);

    return success();
  }

private:
  DenseMap<Value, SmallVector<Value>> &memoryToBanks;
};

// Clean up the empty uses old memory values by either erasing the defining
// operation or replace the block arguments with new ones that corresponds to
// the newly created banks. Change the function signature if the old memory
// values are used as function arguments and/or return values.
LogicalResult cleanUpOldMemRefs(DenseSet<Value> &oldMemRefVals,
                                DenseSet<Operation *> &opsToErase) {
  DenseSet<func::FuncOp> funcsToModify;
  SmallVector<Value, 4> valuesToErase;
  DenseMap<func::FuncOp, SmallVector<unsigned, 4>> erasedArgIndices;
  for (auto &memrefVal : oldMemRefVals) {
    valuesToErase.push_back(memrefVal);
    if (auto blockArg = dyn_cast<BlockArgument>(memrefVal)) {
      if (auto funcOp =
              dyn_cast<func::FuncOp>(blockArg.getOwner()->getParentOp())) {
        funcsToModify.insert(funcOp);
        erasedArgIndices[funcOp].push_back(blockArg.getArgNumber());
      }
    }
  }

  for (auto *op : opsToErase) {
    op->erase();
  }
  // Erase values safely.
  for (auto &memrefVal : valuesToErase) {
    assert(memrefVal.use_empty() && "use must be empty");
    if (auto blockArg = dyn_cast<BlockArgument>(memrefVal)) {
      blockArg.getOwner()->eraseArgument(blockArg.getArgNumber());
    } else if (auto *op = memrefVal.getDefiningOp()) {
      op->erase();
    }
  }

  // Modify the function argument attributes and function type accordingly
  for (auto funcOp : funcsToModify) {
    ArrayAttr existingArgAttrs = funcOp->getAttrOfType<ArrayAttr>("arg_attrs");
    if (existingArgAttrs) {
      SmallVector<Attribute, 4> updatedArgAttrs;
      auto erasedIndices = erasedArgIndices[funcOp];
      DenseSet<unsigned> indicesToErase(erasedIndices.begin(),
                                        erasedIndices.end());
      for (unsigned i = 0; i < existingArgAttrs.size(); ++i) {
        if (!indicesToErase.contains(i))
          updatedArgAttrs.push_back(existingArgAttrs[i]);
      }

      funcOp->setAttr("arg_attrs",
                      ArrayAttr::get(funcOp.getContext(), updatedArgAttrs));
    }

    SmallVector<Type, 4> newArgTypes;
    for (BlockArgument arg : funcOp.getArguments()) {
      newArgTypes.push_back(arg.getType());
    }
    FunctionType newFuncType =
        FunctionType::get(funcOp.getContext(), newArgTypes,
                          funcOp.getFunctionType().getResults());
    funcOp.setType(newFuncType);
  }

  return success();
}

void MemoryBankingPass::runOnOperation() {
  SmallVector<unsigned> bankingFactors(bankingFactorsList.begin(),
                                       bankingFactorsList.end());
  SmallVector<unsigned> bankingDimensions(bankingDimensionsList.begin(),
                                          bankingDimensionsList.end());

  if (getOperation().isExternal() ||
      (bankingFactors.empty() ||
       std::all_of(bankingFactors.begin(), bankingFactors.end(),
                   [](unsigned f) { return f == 1; })))
    return;

  if (std::any_of(bankingFactors.begin(), bankingFactors.end(),
                  [](int f) { return f == 0; })) {
    getOperation().emitError("banking factor must be greater than 1");
    signalPassFailure();
    return;
  }

  if (bankingFactors.size() != bankingDimensions.size()) {
    // For the second check, if banking dimensions are not specified and there
    // is only one banking factor specified, we'll leave it with the default
    // behavior; otherwise, it's an error.
    if (!bankingDimensions.empty() || bankingFactors.size() >= 2) {
      getOperation().emitError("the number of banking factors must be equal to "
                               "the number of banking dimensions");
      signalPassFailure();
      return;
    }
  }

  getOperation().walk([&](mlir::affine::AffineParallelOp parOp) {
    DenseSet<Value> memrefsInPar = collectMemRefs(parOp);

    for (auto memrefVal : memrefsInPar) {
      auto [it, inserted] = memoryToBanks.insert(
          std::make_pair(memrefVal, DenseMap<unsigned, SmallVector<Value>>{}));
      if (inserted)
        it->second = createBanks(memrefVal, bankingFactors, bankingDimensions);
    }
  });

  auto *ctx = &getContext();
  RewritePatternSet patterns(ctx);

  DenseSet<Operation *> processedOps;
  patterns.add<BankAffineLoadPattern>(ctx, bankingFactors, bankingDimensions,
                                      memoryToBanks, oldMemRefVals);
  patterns.add<BankAffineStorePattern>(ctx, bankingFactors, bankingDimensions,
                                       memoryToBanks, opsToErase, processedOps,
                                       oldMemRefVals);
  patterns.add<BankReturnPattern>(ctx, memoryToBanks);

  GreedyRewriteConfig config;
  config.strictMode = GreedyRewriteStrictness::ExistingOps;
  if (failed(
          applyPatternsGreedily(getOperation(), std::move(patterns), config))) {
    signalPassFailure();
  }

  // Clean up the old memref values
  if (failed(cleanUpOldMemRefs(oldMemRefVals, opsToErase))) {
    signalPassFailure();
  }
}

namespace circt {
std::unique_ptr<mlir::Pass>
createMemoryBankingPass(ArrayRef<unsigned> bankingFactors,
                        ArrayRef<unsigned> bankingDimensions) {
  return std::make_unique<MemoryBankingPass>(bankingFactors, bankingDimensions);
}
} // namespace circt
