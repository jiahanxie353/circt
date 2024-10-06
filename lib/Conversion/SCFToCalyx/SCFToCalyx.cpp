//===- SCFToCalyx.cpp - SCF to Calyx pass entry point -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the main SCF to Calyx conversion pass implementation.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/SCFToCalyx.h"
#include "circt/Dialect/Calyx/CalyxHelpers.h"
#include "circt/Dialect/Calyx/CalyxLoweringUtils.h"
#include "circt/Dialect/Calyx/CalyxOps.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Support/LLVM.h"
#include "circt/Transforms/Passes.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <fstream>
#include <optional>
#include <variant>

namespace circt {
#define GEN_PASS_DEF_SCFTOCALYX
#include "circt/Conversion/Passes.h.inc"
} // namespace circt

using namespace llvm;
using namespace mlir;
using namespace mlir::arith;
using namespace mlir::cf;
using namespace mlir::func;
namespace circt {
class ComponentLoweringStateInterface;
namespace scftocalyx {

using json = nlohmann::ordered_json;

//===----------------------------------------------------------------------===//
// Utility types
//===----------------------------------------------------------------------===//

class ScfWhileOp : public calyx::WhileOpInterface<scf::WhileOp> {
public:
  explicit ScfWhileOp(scf::WhileOp op)
      : calyx::WhileOpInterface<scf::WhileOp>(op) {}

  Block::BlockArgListType getBodyArgs() override {
    return getOperation().getAfterArguments();
  }

  Block *getBodyBlock() override { return &getOperation().getAfter().front(); }

  Block *getConditionBlock() override {
    return &getOperation().getBefore().front();
  }

  Value getConditionValue() override {
    return getOperation().getConditionOp().getOperand(0);
  }

  std::optional<int64_t> getBound() override { return std::nullopt; }
};

class ScfForOp : public calyx::RepeatOpInterface<scf::ForOp> {
public:
  explicit ScfForOp(scf::ForOp op) : calyx::RepeatOpInterface<scf::ForOp>(op) {}

  Block::BlockArgListType getBodyArgs() override {
    return getOperation().getRegion().getArguments();
  }

  Block *getBodyBlock() override {
    return &getOperation().getRegion().getBlocks().front();
  }

  std::optional<int64_t> getBound() override {
    return constantTripCount(getOperation().getLowerBound(),
                             getOperation().getUpperBound(),
                             getOperation().getStep());
  }
};

//===----------------------------------------------------------------------===//
// Lowering state classes
//===----------------------------------------------------------------------===//

struct IfScheduleable {
  scf::IfOp ifOp;
};

struct WhileScheduleable {
  /// While operation to schedule.
  ScfWhileOp whileOp;
};

struct ForScheduleable {
  /// For operation to schedule.
  ScfForOp forOp;
  /// Bound
  uint64_t bound;
};

struct CallScheduleable {
  /// Instance for invoking.
  calyx::InstanceOp instanceOp;
  // CallOp for getting the arguments.
  func::CallOp callOp;
};

/// A variant of types representing scheduleable operations.
using Scheduleable =
    std::variant<calyx::GroupOp, WhileScheduleable, ForScheduleable,
                 IfScheduleable, CallScheduleable>;

class IfLoweringStateInterface {
public:
  void setCondReg(scf::IfOp op, calyx::RegisterOp regOp) {
    Operation *operation = op.getOperation();
    assert(condReg.count(operation) == 0 &&
           "A condition register was already set for this scf::IfOp!\n");
    condReg[operation] = regOp;
  }

  calyx::RegisterOp getCondReg(scf::IfOp op) {
    auto it = condReg.find(op.getOperation());
    if (it != condReg.end())
      return it->second;
    return nullptr;
  }

  void setThenGroup(scf::IfOp op, calyx::GroupOp group) {
    Operation *operation = op.getOperation();
    assert(thenGroup.count(operation) == 0 &&
           "A then group was already set for this scf::IfOp!\n");
    thenGroup[operation] = group;
  }

  calyx::GroupOp getThenGroup(scf::IfOp op) {
    auto it = thenGroup.find(op.getOperation());
    assert(it != thenGroup.end() &&
           "No then group was set for this scf::IfOp!\n");
    return it->second;
  }

  void setElseGroup(scf::IfOp op, calyx::GroupOp group) {
    Operation *operation = op.getOperation();
    assert(elseGroup.count(operation) == 0 &&
           "An else group was already set for this scf::IfOp!\n");
    elseGroup[operation] = group;
  }

  calyx::GroupOp getElseGroup(scf::IfOp op) {
    auto it = elseGroup.find(op.getOperation());
    assert(it != elseGroup.end() &&
           "No else group was set for this scf::IfOp!\n");
    return it->second;
  }

  void setResultRegs(scf::IfOp op, calyx::RegisterOp reg, unsigned idx) {
    assert(resultRegs[op.getOperation()].count(idx) == 0 &&
           "A register was already registered for the given yield result.\n");
    assert(idx < op->getNumOperands());
    resultRegs[op.getOperation()][idx] = reg;
  }

  const DenseMap<unsigned, calyx::RegisterOp> &getResultRegs(scf::IfOp op) {
    return resultRegs[op.getOperation()];
  }

  calyx::RegisterOp getResultRegs(scf::IfOp op, unsigned idx) {
    auto regs = getResultRegs(op);
    auto it = regs.find(idx);
    assert(it != regs.end() && "resultReg not found");
    return it->second;
  }

private:
  DenseMap<Operation *, calyx::RegisterOp> condReg;
  DenseMap<Operation *, calyx::GroupOp> thenGroup;
  DenseMap<Operation *, calyx::GroupOp> elseGroup;
  DenseMap<Operation *, DenseMap<unsigned, calyx::RegisterOp>> resultRegs;
};

class WhileLoopLoweringStateInterface
    : calyx::LoopLoweringStateInterface<ScfWhileOp> {
public:
  SmallVector<calyx::GroupOp> getWhileLoopInitGroups(ScfWhileOp op) {
    return getLoopInitGroups(std::move(op));
  }
  calyx::GroupOp buildWhileLoopIterArgAssignments(
      OpBuilder &builder, ScfWhileOp op, calyx::ComponentOp componentOp,
      Twine uniqueSuffix, MutableArrayRef<OpOperand> ops) {
    return buildLoopIterArgAssignments(builder, std::move(op), componentOp,
                                       uniqueSuffix, ops);
  }
  void addWhileLoopIterReg(ScfWhileOp op, calyx::RegisterOp reg, unsigned idx) {
    return addLoopIterReg(std::move(op), reg, idx);
  }
  const DenseMap<unsigned, calyx::RegisterOp> &
  getWhileLoopIterRegs(ScfWhileOp op) {
    return getLoopIterRegs(std::move(op));
  }
  void setWhileLoopLatchGroup(ScfWhileOp op, calyx::GroupOp group) {
    return setLoopLatchGroup(std::move(op), group);
  }
  calyx::GroupOp getWhileLoopLatchGroup(ScfWhileOp op) {
    return getLoopLatchGroup(std::move(op));
  }
  void setWhileLoopInitGroups(ScfWhileOp op,
                              SmallVector<calyx::GroupOp> groups) {
    return setLoopInitGroups(std::move(op), std::move(groups));
  }
};

class ForLoopLoweringStateInterface
    : calyx::LoopLoweringStateInterface<ScfForOp> {
public:
  SmallVector<calyx::GroupOp> getForLoopInitGroups(ScfForOp op) {
    return getLoopInitGroups(std::move(op));
  }
  calyx::GroupOp buildForLoopIterArgAssignments(
      OpBuilder &builder, ScfForOp op, calyx::ComponentOp componentOp,
      Twine uniqueSuffix, MutableArrayRef<OpOperand> ops) {
    return buildLoopIterArgAssignments(builder, std::move(op), componentOp,
                                       uniqueSuffix, ops);
  }
  void addForLoopIterReg(ScfForOp op, calyx::RegisterOp reg, unsigned idx) {
    return addLoopIterReg(std::move(op), reg, idx);
  }
  const DenseMap<unsigned, calyx::RegisterOp> &getForLoopIterRegs(ScfForOp op) {
    return getLoopIterRegs(std::move(op));
  }
  calyx::RegisterOp getForLoopIterReg(ScfForOp op, unsigned idx) {
    return getLoopIterReg(std::move(op), idx);
  }
  void setForLoopLatchGroup(ScfForOp op, calyx::GroupOp group) {
    return setLoopLatchGroup(std::move(op), group);
  }
  calyx::GroupOp getForLoopLatchGroup(ScfForOp op) {
    return getLoopLatchGroup(std::move(op));
  }
  void setForLoopInitGroups(ScfForOp op, SmallVector<calyx::GroupOp> groups) {
    return setLoopInitGroups(std::move(op), std::move(groups));
  }
};

class PipeOpLoweringStateInterface {
public:
  void setPipeResReg(Operation *op, calyx::RegisterOp reg) {
    assert(isa<calyx::MultPipeLibOp>(op) || isa<calyx::DivUPipeLibOp>(op) ||
           isa<calyx::DivSPipeLibOp>(op) || isa<calyx::RemUPipeLibOp>(op) ||
           isa<calyx::RemSPipeLibOp>(op));
    assert(resultRegs.count(op) == 0 &&
           "A register was already set for this pipe operation!\n");
    resultRegs[op] = reg;
  }
  // Get the register for a specific pipe operation
  calyx::RegisterOp getPipeResReg(Operation *op) {
    auto it = resultRegs.find(op);
    assert(it != resultRegs.end() &&
           "No register was set for this pipe operation!\n");
    return it->second;
  }

private:
  DenseMap<Operation *, calyx::RegisterOp> resultRegs;
};

/// Handles the current state of lowering of a Calyx component. It is mainly
/// used as a key/value store for recording information during partial lowering,
/// which is required at later lowering passes.
class ComponentLoweringState : public calyx::ComponentLoweringStateInterface,
                               public WhileLoopLoweringStateInterface,
                               public ForLoopLoweringStateInterface,
                               public IfLoweringStateInterface,
                               public PipeOpLoweringStateInterface,
                               public calyx::SchedulerInterface<Scheduleable> {
public:
  ComponentLoweringState(calyx::ComponentOp component)
      : calyx::ComponentLoweringStateInterface(component) {}
};

//===----------------------------------------------------------------------===//
// Conversion patterns
//===----------------------------------------------------------------------===//

/// Iterate through the operations of a source function and instantiate
/// components or primitives based on the type of the operations.
class BuildOpGroups : public calyx::FuncOpPartialLoweringPattern {
  using FuncOpPartialLoweringPattern::FuncOpPartialLoweringPattern;

  LogicalResult
  partiallyLowerFuncToComp(FuncOp funcOp,
                           PatternRewriter &rewriter) const override {
    /// We walk the operations of the funcOp to ensure that all def's have
    /// been visited before their uses.
    bool opBuiltSuccessfully = true;
    funcOp.walk([&](Operation *_op) {
      opBuiltSuccessfully &=
          TypeSwitch<mlir::Operation *, bool>(_op)
              .template Case<arith::ConstantOp, ReturnOp, BranchOpInterface,
                             /// SCF
                             scf::YieldOp, scf::WhileOp, scf::ForOp, scf::IfOp,
                             /// memref
                             memref::AllocOp, memref::AllocaOp, memref::LoadOp,
                             memref::StoreOp, memref::GetGlobalOp,
                             /// standard arithmetic
                             AddIOp, SubIOp, CmpIOp, ShLIOp, ShRUIOp, ShRSIOp,
                             AndIOp, XOrIOp, OrIOp, ExtUIOp, ExtSIOp, TruncIOp,
                             MulIOp, DivUIOp, DivSIOp, RemUIOp, RemSIOp,
                             /// floating point
                             AddFOp, MulFOp, CmpFOp,
                             /// other
                             SelectOp, IndexCastOp, CallOp>(
                  [&](auto op) { return buildOp(rewriter, op).succeeded(); })
              .template Case<FuncOp, scf::ConditionOp>([&](auto) {
                /// Skip: these special cases will be handled separately.
                return true;
              })
              .Default([&](auto op) {
                op->emitError() << "Unhandled operation during BuildOpGroups()";
                return false;
              });

      return opBuiltSuccessfully ? WalkResult::advance()
                                 : WalkResult::interrupt();
    });

    std::ofstream outFile("data.json");
    if (outFile.is_open()) {
      outFile << getState<ComponentLoweringState>().getExtMemData().dump(2);
      outFile.close();
    } else {
      llvm::errs() << "Unable to open file for writing\n";
    }

    getState<ComponentLoweringState>().getComponentOp().dump();
    return success(opBuiltSuccessfully);
  }

private:
  /// Op builder specializations.
  LogicalResult buildOp(PatternRewriter &rewriter, scf::YieldOp yieldOp) const;
  LogicalResult buildOp(PatternRewriter &rewriter,
                        BranchOpInterface brOp) const;
  LogicalResult buildOp(PatternRewriter &rewriter,
                        arith::ConstantOp constOp) const;
  LogicalResult buildOp(PatternRewriter &rewriter, SelectOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, AddIOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, SubIOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, MulIOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, DivUIOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, DivSIOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, RemUIOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, RemSIOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, AddFOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, MulFOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, CmpFOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, ShRUIOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, ShRSIOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, ShLIOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, AndIOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, OrIOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, XOrIOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, CmpIOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, TruncIOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, ExtUIOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, ExtSIOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, ReturnOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, IndexCastOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, memref::AllocOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, memref::AllocaOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter,
                        memref::GetGlobalOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, memref::LoadOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, memref::StoreOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, scf::WhileOp whileOp) const;
  LogicalResult buildOp(PatternRewriter &rewriter, scf::ForOp forOp) const;
  LogicalResult buildOp(PatternRewriter &rewriter, scf::IfOp ifOp) const;
  LogicalResult buildOp(PatternRewriter &rewriter, CallOp callOp) const;

  /// buildLibraryOp will build a TCalyxLibOp inside a TGroupOp based on the
  /// source operation TSrcOp.
  template <typename TGroupOp, typename TCalyxLibOp, typename TSrcOp>
  LogicalResult buildLibraryOp(PatternRewriter &rewriter, TSrcOp op,
                               TypeRange srcTypes, TypeRange dstTypes,
                               calyx::RegisterOp srcReg = nullptr,
                               calyx::RegisterOp dstReg = nullptr) const {
    auto isPipeLibOp = [](Value val) -> bool {
      if (Operation *defOp = val.getDefiningOp()) {
        return isa<calyx::MultPipeLibOp, calyx::DivUPipeLibOp,
                   calyx::DivSPipeLibOp, calyx::RemUPipeLibOp,
                   calyx::RemSPipeLibOp>(defOp);
      }
      return false;
    };

    assert((srcReg && dstReg) || (!srcReg && !dstReg));
    bool isSequential = srcReg && dstReg;

    SmallVector<Type> types;
    llvm::append_range(types, srcTypes);
    llvm::append_range(types, dstTypes);

    auto calyxOp =
        getState<ComponentLoweringState>().getNewLibraryOpInstance<TCalyxLibOp>(
            rewriter, op.getLoc(), types);

    auto directions = calyxOp.portDirections();
    SmallVector<Value, 4> opInputPorts;
    SmallVector<Value, 4> opOutputPorts;
    for (auto dir : enumerate(directions)) {
      if (dir.value() == calyx::Direction::Input)
        opInputPorts.push_back(calyxOp.getResult(dir.index()));
      else
        opOutputPorts.push_back(calyxOp.getResult(dir.index()));
    }
    assert(
        opInputPorts.size() == op->getNumOperands() &&
        opOutputPorts.size() == op->getNumResults() &&
        "Expected an equal number of in/out ports in the Calyx library op with "
        "respect to the number of operands/results of the source operation.");

    /// Create assignments to the inputs of the library op.
    auto group = createGroupForOp<TGroupOp>(rewriter, op);

    if (isSequential) {
      auto groupOp = cast<calyx::GroupOp>(group);
      getState<ComponentLoweringState>().addBlockScheduleable(op->getBlock(),
                                                              groupOp);
    }

    rewriter.setInsertionPointToEnd(group.getBodyBlock());

    for (auto dstOp : enumerate(opInputPorts)) {
      if (isPipeLibOp(dstOp.value()))
        rewriter.create<calyx::AssignOp>(op.getLoc(), dstOp.value(),
                                         srcReg.getOut());
      else
        rewriter.create<calyx::AssignOp>(op.getLoc(), dstOp.value(),
                                         op->getOperand(dstOp.index()));
    }

    /// Replace the result values of the source operator with the new operator.
    for (auto res : enumerate(opOutputPorts)) {
      getState<ComponentLoweringState>().registerEvaluatingGroup(res.value(),
                                                                 group);
      if (isSequential)
        op->getResult(res.index()).replaceAllUsesWith(dstReg.getOut());
      else
        op->getResult(res.index()).replaceAllUsesWith(res.value());
    }

    if (isSequential) {
      auto groupOp = cast<calyx::GroupOp>(group);
      buildAssignmentsForRegisterWrite(
          rewriter, groupOp,
          getState<ComponentLoweringState>().getComponentOp(), dstReg,
          calyxOp.getOut());
    }

    return success();
  }

  /// buildLibraryOp which provides in- and output types based on the operands
  /// and results of the op argument.
  template <typename TGroupOp, typename TCalyxLibOp, typename TSrcOp>
  LogicalResult buildLibraryOp(PatternRewriter &rewriter, TSrcOp op,
                               calyx::RegisterOp srcReg = nullptr,
                               calyx::RegisterOp dstReg = nullptr) const {
    return buildLibraryOp<TGroupOp, TCalyxLibOp, TSrcOp>(
        rewriter, op, op.getOperandTypes(), op->getResultTypes(), srcReg,
        dstReg);
  }

  /// Creates a group named by the basic block which the input op resides in.
  template <typename TGroupOp>
  TGroupOp createGroupForOp(PatternRewriter &rewriter, Operation *op) const {
    Block *block = op->getBlock();
    auto groupName = getState<ComponentLoweringState>().getUniqueName(
        loweringState().blockName(block));
    return calyx::createGroup<TGroupOp>(
        rewriter, getState<ComponentLoweringState>().getComponentOp(),
        op->getLoc(), groupName);
  }

  /// buildLibraryBinaryPipeOp will build a TCalyxLibBinaryPipeOp, to
  /// deal with MulIOp, DivUIOp and RemUIOp.
  template <typename TOpType, typename TSrcOp>
  LogicalResult buildLibraryBinaryPipeOp(PatternRewriter &rewriter, TSrcOp op,
                                         TOpType opPipe, Value out) const {
    llvm::errs() << "before building:\n";
    getState<ComponentLoweringState>().getComponentOp().dump();
    StringRef opName = TSrcOp::getOperationName().split(".").second;
    Location loc = op.getLoc();
    Type width = op.getResult().getType();
    // Pass the result from the Operation to the Calyx primitive.
    op.getResult().replaceAllUsesWith(out);
    auto reg = createRegister(
        op.getLoc(), rewriter, getComponent(), width,
        getState<ComponentLoweringState>().getUniqueName(opName));

    // Operation pipelines are not combinational, so a GroupOp is required.
    auto group = createGroupForOp<calyx::GroupOp>(rewriter, op);
    OpBuilder builder(group->getRegion(0));
    getState<ComponentLoweringState>().addBlockScheduleable(op->getBlock(),
                                                            group);

    rewriter.setInsertionPointToEnd(group.getBodyBlock());
    rewriter.create<calyx::AssignOp>(loc, opPipe.getLeft(), op.getLhs());
    rewriter.create<calyx::AssignOp>(loc, opPipe.getRight(), op.getRhs());
    // Write the output to this register.
    rewriter.create<calyx::AssignOp>(loc, reg.getIn(), out);
    // The write enable port is high when the pipeline is done.
    rewriter.create<calyx::AssignOp>(loc, reg.getWriteEn(), opPipe.getDone());
    // Set pipelineOp to high as long as its done signal is not high.
    // This prevents the pipelineOP from executing for the cycle that we write
    // to register. To get !(pipelineOp.done) we do 1 xor pipelineOp.done
    hw::ConstantOp c1 = createConstant(loc, rewriter, getComponent(), 1, 1);
    rewriter.create<calyx::AssignOp>(
        loc, opPipe.getGo(), c1,
        comb::createOrFoldNot(group.getLoc(), opPipe.getDone(), builder));
    // The group is done when the register write is complete.
    rewriter.create<calyx::GroupDoneOp>(loc, reg.getDone());

    if (isa<calyx::AddFNOp>(opPipe)) {
      auto opFN = cast<calyx::AddFNOp>(opPipe);
      hw::ConstantOp subOp;
      if (isa<arith::AddFOp>(op)) {
        subOp = createConstant(loc, rewriter, getComponent(), 1, 0);
      } else {
        subOp = createConstant(loc, rewriter, getComponent(), 1, 1);
      }
      rewriter.create<calyx::AssignOp>(loc, opFN.getSubOp(), subOp);
    }

    // Register the values for the pipeline.
    getState<ComponentLoweringState>().registerEvaluatingGroup(out, group);
    getState<ComponentLoweringState>().registerEvaluatingGroup(opPipe.getLeft(),
                                                               group);
    getState<ComponentLoweringState>().registerEvaluatingGroup(
        opPipe.getRight(), group);

    getState<ComponentLoweringState>().setPipeResReg(out.getDefiningOp(), reg);

    return success();
  }

  /// Creates assignments within the provided group to the address ports of the
  /// memoryOp based on the provided addressValues.
  void assignAddressPorts(PatternRewriter &rewriter, Location loc,
                          calyx::GroupInterface group,
                          calyx::MemoryInterface memoryInterface,
                          Operation::operand_range addressValues) const {
    IRRewriter::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToEnd(group.getBody());
    auto addrPorts = memoryInterface.addrPorts();
    if (addressValues.empty()) {
      assert(
          addrPorts.size() == 1 &&
          "We expected a 1 dimensional memory of size 1 because there were no "
          "address assignment values");
      // Assign to address 1'd0 in memory.
      rewriter.create<calyx::AssignOp>(
          loc, addrPorts[0],
          createConstant(loc, rewriter, getComponent(), 1, 0));
    } else {
      assert(addrPorts.size() == addressValues.size() &&
             "Mismatch between number of address ports of the provided memory "
             "and address assignment values");
      for (auto address : enumerate(addressValues))
        rewriter.create<calyx::AssignOp>(loc, addrPorts[address.index()],
                                         address.value());
    }
  }
};

LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     memref::LoadOp loadOp) const {
  Value memref = loadOp.getMemref();
  auto memoryInterface =
      getState<ComponentLoweringState>().getMemoryInterface(memref);
  auto group = createGroupForOp<calyx::GroupOp>(rewriter, loadOp);
  assignAddressPorts(rewriter, loadOp.getLoc(), group, memoryInterface,
                     loadOp.getIndices());

  rewriter.setInsertionPointToEnd(group.getBodyBlock());

  bool needReg = true;
  Value res;
  Value regWriteEn =
      createConstant(loadOp.getLoc(), rewriter, getComponent(), 1, 1);
  if (memoryInterface.readEnOpt().has_value()) {
    auto oneI1 =
        calyx::createConstant(loadOp.getLoc(), rewriter, getComponent(), 1, 1);
    rewriter.create<calyx::AssignOp>(loadOp.getLoc(), memoryInterface.readEn(),
                                     oneI1);
    regWriteEn = memoryInterface.done();
    if (calyx::noStoresToMemory(memref) &&
        calyx::singleLoadFromMemory(memref)) {
      // Single load from memory; we do not need to write the output to a
      // register. The readData value will be held until readEn is asserted
      // again
      needReg = false;
      rewriter.create<calyx::GroupDoneOp>(loadOp.getLoc(),
                                          memoryInterface.done());
      // We refrain from replacing the loadOp result with
      // memoryInterface.readData, since multiple loadOp's need to be converted
      // to a single memory's ReadData. If this replacement is done now, we lose
      // the link between which SSA memref::LoadOp values map to which groups
      // for loading a value from the Calyx memory. At this point of lowering,
      // we keep the memref::LoadOp SSA value, and do value replacement _after_
      // control has been generated (see LateSSAReplacement). This is *vital*
      // for things such as calyx::InlineCombGroups to be able to properly track
      // which memory assignment groups belong to which accesses.
      res = loadOp.getResult();
    }
  } else if (memoryInterface.contentEnOpt().has_value()) {
    auto oneI1 =
        calyx::createConstant(loadOp.getLoc(), rewriter, getComponent(), 1, 1);
    auto zeroI1 =
        calyx::createConstant(loadOp.getLoc(), rewriter, getComponent(), 1, 0);
    rewriter.create<calyx::AssignOp>(loadOp.getLoc(),
                                     memoryInterface.contentEn(), oneI1);
    rewriter.create<calyx::AssignOp>(loadOp.getLoc(), memoryInterface.writeEn(),
                                     zeroI1);

    // Writing to calyx.seq_mem even when content_en = 1 and write_en = 0.
    // This is because calyx.seq_mem has write_together attribute for write_en
    // and write_data. But since the value of write_data doesn't matter, we just
    // create bitvector with 0's.
    if (memoryInterface.isSeqMem()) {
      auto memOperand = llvm::cast<MemRefType>(loadOp.getOperand(0).getType());
      Value writeZero;
      if (isa<FloatType>(memOperand.getElementType())) {
        auto floatType = cast<FloatType>(memOperand.getElementType());
        auto wrZeroFlt = rewriter.getFloatAttr(
            memOperand.getElementType(),
            APFloat::getZero(floatType.getFloatSemantics()));
        writeZero =
            rewriter.create<calyx::ConstantOp>(loadOp.getLoc(), wrZeroFlt);
      } else {
        auto wrZeroInt = rewriter.getIntegerAttr(
            memOperand.getElementType(),
            APInt::getZero(
                memOperand.getElementType().getIntOrFloatBitWidth()));
        writeZero = rewriter.create<hw::ConstantOp>(loadOp.getLoc(), wrZeroInt);
      }
      rewriter.create<calyx::AssignOp>(loadOp.getLoc(),
                                       memoryInterface.writeData(), writeZero);
    }

    regWriteEn = memoryInterface.done();
    if (calyx::noStoresToMemory(memref) &&
        calyx::singleLoadFromMemory(memref) && !memoryInterface.isSeqMem()) {
      // Single load from memory; we do not need to write the output to a
      // register. The readData value will be held until contentEn is asserted
      // again
      needReg = false;
      rewriter.create<calyx::GroupDoneOp>(loadOp.getLoc(),
                                          memoryInterface.done());
      // We refrain from replacing the loadOp result with
      // memoryInterface.readData, since multiple loadOp's need to be converted
      // to a single memory's ReadData. If this replacement is done now, we lose
      // the link between which SSA memref::LoadOp values map to which groups
      // for loading a value from the Calyx memory. At this point of lowering,
      // we keep the memref::LoadOp SSA value, and do value replacement _after_
      // control has been generated (see LateSSAReplacement). This is *vital*
      // for things such as calyx::InlineCombGroups to be able to properly track
      // which memory assignment groups belong to which accesses.
      res = loadOp.getResult();
    }
  }

  if (needReg) {
    // Multiple loads from the same memory; In this case, we _may_ have a
    // structural hazard in the design we generate. To get around this, we
    // conservatively place a register in front of each load operation, and
    // replace all uses of the loaded value with the register output. Reading
    // for sequential memories will cause a read to take at least 2 cycles,
    // but it will usually be better because combinational reads on memories
    // can significantly decrease the maximum achievable frequency.
    auto reg = createRegister(
        loadOp.getLoc(), rewriter, getComponent(),
        loadOp.getMemRefType().getElementType(),
        getState<ComponentLoweringState>().getUniqueName("load"));
    rewriter.setInsertionPointToEnd(group.getBodyBlock());
    rewriter.create<calyx::AssignOp>(loadOp.getLoc(), reg.getIn(),
                                     memoryInterface.readData());
    rewriter.create<calyx::AssignOp>(loadOp.getLoc(), reg.getWriteEn(),
                                     regWriteEn);
    rewriter.create<calyx::GroupDoneOp>(loadOp.getLoc(), reg.getDone());
    loadOp.getResult().replaceAllUsesWith(reg.getOut());
    res = reg.getOut();
  }

  getState<ComponentLoweringState>().registerEvaluatingGroup(res, group);
  getState<ComponentLoweringState>().addBlockScheduleable(loadOp->getBlock(),
                                                          group);
  return success();
}

LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     memref::StoreOp storeOp) const {
  auto memoryInterface = getState<ComponentLoweringState>().getMemoryInterface(
      storeOp.getMemref());
  auto group = createGroupForOp<calyx::GroupOp>(rewriter, storeOp);

  // This is a sequential group, so register it as being scheduleable for the
  // block.
  getState<ComponentLoweringState>().addBlockScheduleable(storeOp->getBlock(),
                                                          group);
  assignAddressPorts(rewriter, storeOp.getLoc(), group, memoryInterface,
                     storeOp.getIndices());
  rewriter.setInsertionPointToEnd(group.getBodyBlock());
  rewriter.create<calyx::AssignOp>(
      storeOp.getLoc(), memoryInterface.writeData(), storeOp.getValueToStore());
  rewriter.create<calyx::AssignOp>(
      storeOp.getLoc(), memoryInterface.writeEn(),
      createConstant(storeOp.getLoc(), rewriter, getComponent(), 1, 1));
  if (memoryInterface.contentEnOpt().has_value()) {
    // If memory has content enable, it must be asserted when writing
    rewriter.create<calyx::AssignOp>(
        storeOp.getLoc(), memoryInterface.contentEn(),
        createConstant(storeOp.getLoc(), rewriter, getComponent(), 1, 1));
  }
  rewriter.create<calyx::GroupDoneOp>(storeOp.getLoc(), memoryInterface.done());

  return success();
}

LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     MulIOp mul) const {
  Location loc = mul.getLoc();
  Type width = mul.getResult().getType(), one = rewriter.getI1Type();
  auto mulPipe =
      getState<ComponentLoweringState>()
          .getNewLibraryOpInstance<calyx::MultPipeLibOp>(
              rewriter, loc, {one, one, one, width, width, width, one});
  return buildLibraryBinaryPipeOp<calyx::MultPipeLibOp>(
      rewriter, mul, mulPipe,
      /*out=*/mulPipe.getOut());
}

LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     DivUIOp div) const {
  Location loc = div.getLoc();
  Type width = div.getResult().getType(), one = rewriter.getI1Type();
  auto divPipe =
      getState<ComponentLoweringState>()
          .getNewLibraryOpInstance<calyx::DivUPipeLibOp>(
              rewriter, loc, {one, one, one, width, width, width, one});
  return buildLibraryBinaryPipeOp<calyx::DivUPipeLibOp>(
      rewriter, div, divPipe,
      /*out=*/divPipe.getOut());
}

LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     DivSIOp div) const {
  Location loc = div.getLoc();
  Type width = div.getResult().getType(), one = rewriter.getI1Type();
  auto divPipe =
      getState<ComponentLoweringState>()
          .getNewLibraryOpInstance<calyx::DivSPipeLibOp>(
              rewriter, loc, {one, one, one, width, width, width, one});
  return buildLibraryBinaryPipeOp<calyx::DivSPipeLibOp>(
      rewriter, div, divPipe,
      /*out=*/divPipe.getOut());
}

LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     RemUIOp rem) const {
  Location loc = rem.getLoc();
  Type width = rem.getResult().getType(), one = rewriter.getI1Type();
  auto remPipe =
      getState<ComponentLoweringState>()
          .getNewLibraryOpInstance<calyx::RemUPipeLibOp>(
              rewriter, loc, {one, one, one, width, width, width, one});
  return buildLibraryBinaryPipeOp<calyx::RemUPipeLibOp>(
      rewriter, rem, remPipe,
      /*out=*/remPipe.getOut());
}

LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     RemSIOp rem) const {
  Location loc = rem.getLoc();
  Type width = rem.getResult().getType(), one = rewriter.getI1Type();
  auto remPipe =
      getState<ComponentLoweringState>()
          .getNewLibraryOpInstance<calyx::RemSPipeLibOp>(
              rewriter, loc, {one, one, one, width, width, width, one});
  return buildLibraryBinaryPipeOp<calyx::RemSPipeLibOp>(
      rewriter, rem, remPipe,
      /*out=*/remPipe.getOut());
}

LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     AddFOp addf) const {
  Location loc = addf.getLoc();
  Type width = addf.getResult().getType();
  IntegerType one = rewriter.getI1Type(), three = rewriter.getIntegerType(3),
              five = rewriter.getIntegerType(5);
  auto addFN =
      getState<ComponentLoweringState>()
          .getNewLibraryOpInstance<calyx::AddFNOp>(
              rewriter, loc,
              {one, one, one, one, one, width, width, three, width, five, one});
  return buildLibraryBinaryPipeOp<calyx::AddFNOp>(rewriter, addf, addFN,
                                                  addFN.getOut());
}

LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     MulFOp mulf) const {
  Location loc = mulf.getLoc();
  Type width = mulf.getResult().getType();
  IntegerType one = rewriter.getI1Type(), three = rewriter.getIntegerType(3),
              five = rewriter.getIntegerType(5);
  auto mulFN =
      getState<ComponentLoweringState>()
          .getNewLibraryOpInstance<calyx::MulFNOp>(
              rewriter, loc,
              {one, one, one, one, width, width, three, width, five, one});
  return buildLibraryBinaryPipeOp<calyx::MulFNOp>(rewriter, mulf, mulFN,
                                                  mulFN.getOut());
}

LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     CmpFOp op) const {
  switch (op.getPredicate()) {
  case CmpFPredicate::UEQ:
  case CmpFPredicate::OEQ:
    return buildLibraryOp<calyx::CombGroupOp, calyx::EqLibOp>(rewriter, op);
  case CmpFPredicate::UNE:
  case CmpFPredicate::ONE:
    return buildLibraryOp<calyx::CombGroupOp, calyx::NeqLibOp>(rewriter, op);
  case CmpFPredicate::UGE:
  case CmpFPredicate::OGE:
    return buildLibraryOp<calyx::CombGroupOp, calyx::SgeLibOp>(rewriter, op);
  case CmpFPredicate::ULT:
  case CmpFPredicate::OLT:
    return buildLibraryOp<calyx::CombGroupOp, calyx::SltLibOp>(rewriter, op);
  case CmpFPredicate::UGT:
  case CmpFPredicate::OGT:
    return buildLibraryOp<calyx::CombGroupOp, calyx::SgtLibOp>(rewriter, op);
  case CmpFPredicate::ULE:
  case CmpFPredicate::OLE:
    return buildLibraryOp<calyx::CombGroupOp, calyx::SleLibOp>(rewriter, op);
  default:
    llvm_unreachable("unexpected comparison predicate");
  }
}

template <typename TAllocOp>
static LogicalResult buildAllocOp(ComponentLoweringState &componentState,
                                  PatternRewriter &rewriter, TAllocOp allocOp) {
  rewriter.setInsertionPointToStart(
      componentState.getComponentOp().getBodyBlock());
  MemRefType memtype = allocOp.getType();
  SmallVector<int64_t> addrSizes;
  SmallVector<int64_t> sizes;
  for (int64_t dim : memtype.getShape()) {
    sizes.push_back(dim);
    addrSizes.push_back(calyx::handleZeroWidth(dim));
  }
  // If memref has no size (e.g., memref<i32>) create a 1 dimensional memory of
  // size 1.
  if (sizes.empty() && addrSizes.empty()) {
    sizes.push_back(1);
    addrSizes.push_back(1);
  }
  auto memoryOp = rewriter.create<calyx::MemoryOp>(
      allocOp.getLoc(), componentState.getUniqueName("mem"),
      memtype.getElementType(), sizes, addrSizes);

  // Externalize memories conditionally (only in the top-level component because
  // Calyx compiler requires it as a well-formness check).
  memoryOp->setAttr(
      "external", IntegerAttr::get(rewriter.getI1Type(), llvm::APInt(1, 1)));
  componentState.registerMemoryInterface(allocOp.getResult(),
                                         calyx::MemoryInterface(memoryOp));

  bool isFloat = !memtype.getElementType().isInteger();
  
  auto shape = allocOp.getType().getShape();
  std::vector<int> dimensions;
  int totalSize = 1;
  for (auto dim : shape) {
    totalSize *= dim;
    dimensions.push_back(dim);
  }

  std::vector<double> flattenedVals(totalSize, 0);

  // Helper function to get the correct indices
  auto getIndices = [&dimensions](int flatIndex) {
    std::vector<int> indices(dimensions.size(), 0);
    for (int i = dimensions.size() - 1; i >= 0; --i) {
      indices[i] = flatIndex % dimensions[i];
      flatIndex /= dimensions[i];
    }
    return indices;
  };

  json result = json::array();
  if (isa<memref::GetGlobalOp>(allocOp)) {
    auto getGlobalOp = cast<memref::GetGlobalOp>(allocOp);
    auto *symbolTableOp =
        getGlobalOp->template getParentWithTrait<mlir::OpTrait::SymbolTable>();
    auto globalOp = dyn_cast_or_null<memref::GlobalOp>(
        SymbolTable::lookupSymbolIn(symbolTableOp, getGlobalOp.getNameAttr()));
    // Flatten the values in the attribute
    auto cstAttr = llvm::dyn_cast_or_null<DenseElementsAttr>(
        globalOp.getConstantInitValue());
    int sizeCount = 0;
    for (auto attr : cstAttr.template getValues<Attribute>()) {
      if (auto fltAttr = dyn_cast<mlir::FloatAttr>(attr))
        flattenedVals[sizeCount++] = fltAttr.getValueAsDouble();
      else if (auto intAttr = dyn_cast<mlir::IntegerAttr>(attr))
        flattenedVals[sizeCount++] = intAttr.getInt();
    }

    rewriter.eraseOp(globalOp);
  }
  
  // Put the flattened values in the multi-dimensional structure
  for (size_t i = 0; i < flattenedVals.size(); ++i) {
    std::vector<int> indices = getIndices(i);
    json *nested = &result;
    for (size_t j = 0; j < indices.size() - 1; ++j) {
      while (nested->size() <= static_cast<json::size_type>(indices[j])) {
        nested->push_back(json::array());
      }
      nested = &(*nested)[indices[j]];
    }
    if (isFloat)
      nested->push_back(flattenedVals[i]);
    else
      nested->push_back(static_cast<int64_t>(flattenedVals[i]));
  }

  componentState.setDataField(memoryOp.getName(), result);
  auto width = memtype.getElementType().getIntOrFloatBitWidth();

  std::string numType;
  bool isSigned;
  if (memtype.getElementType().isInteger()) {
    numType = "bitnum";
    isSigned = false;
  } else {
    numType = "floating_point";
    isSigned = true;
  }
  componentState.setFormat(memoryOp.getName(), numType, isSigned, width);

  return success();
}

LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     memref::AllocOp allocOp) const {
  return buildAllocOp(getState<ComponentLoweringState>(), rewriter, allocOp);
}

LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     memref::AllocaOp allocOp) const {
  return buildAllocOp(getState<ComponentLoweringState>(), rewriter, allocOp );
}

LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     memref::GetGlobalOp getGlobalOp) const {
  return buildAllocOp(getState<ComponentLoweringState>(), rewriter, getGlobalOp);
}

LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     scf::YieldOp yieldOp) const {
  if (yieldOp.getOperands().empty() &&
      isa<scf::ForOp>(yieldOp->getParentOp())) {
    // If yield operands are empty, we assume we have a for loop.
    auto forOp = dyn_cast<scf::ForOp>(yieldOp->getParentOp());
    assert(forOp && "Empty yieldOps should only be located within ForOps");
    ScfForOp forOpInterface(forOp);

    // Get the ForLoop's Induction Register.
    auto inductionReg =
        getState<ComponentLoweringState>().getForLoopIterReg(forOpInterface, 0);

    Type regWidth = inductionReg.getOut().getType();
    // Adder should have same width as the inductionReg.
    SmallVector<Type> types(3, regWidth);
    auto addOp = getState<ComponentLoweringState>()
                     .getNewLibraryOpInstance<calyx::AddLibOp>(
                         rewriter, forOp.getLoc(), types);

    auto directions = addOp.portDirections();
    // For an add operation, we expect two input ports and one output port
    SmallVector<Value, 2> opInputPorts;
    Value opOutputPort;
    for (auto dir : enumerate(directions)) {
      switch (dir.value()) {
      case calyx::Direction::Input: {
        opInputPorts.push_back(addOp.getResult(dir.index()));
        break;
      }
      case calyx::Direction::Output: {
        opOutputPort = addOp.getResult(dir.index());
        break;
      }
      }
    }

    // "Latch Group" increments inductionReg by forLoop's step value.
    calyx::ComponentOp componentOp =
        getState<ComponentLoweringState>().getComponentOp();
    SmallVector<StringRef, 4> groupIdentifier = {
        "incr", getState<ComponentLoweringState>().getUniqueName(forOp),
        "induction", "var"};
    auto groupOp = calyx::createGroup<calyx::GroupOp>(
        rewriter, componentOp, forOp.getLoc(),
        llvm::join(groupIdentifier, "_"));
    rewriter.setInsertionPointToEnd(groupOp.getBodyBlock());

    // Assign inductionReg.out to the left port of the adder.
    Value leftOp = opInputPorts.front();
    rewriter.create<calyx::AssignOp>(forOp.getLoc(), leftOp,
                                     inductionReg.getOut());
    // Assign forOp.getConstantStep to the right port of the adder.
    Value rightOp = opInputPorts.back();
    rewriter.create<calyx::AssignOp>(
        forOp.getLoc(), rightOp,
        createConstant(forOp->getLoc(), rewriter, componentOp,
                       regWidth.getIntOrFloatBitWidth(),
                       forOp.getConstantStep().value().getSExtValue()));
    // Assign adder's output port to inductionReg.
    buildAssignmentsForRegisterWrite(rewriter, groupOp, componentOp,
                                     inductionReg, opOutputPort);
    // Set group as For Loop's "latch" group.
    getState<ComponentLoweringState>().setForLoopLatchGroup(forOpInterface,
                                                            groupOp);
    getState<ComponentLoweringState>().registerEvaluatingGroup(opOutputPort,
                                                               groupOp);
    return success();
  }
  // If yieldOp for a for loop is not empty, then we do not transform for loop.
  if (dyn_cast<scf::ForOp>(yieldOp->getParentOp())) {
    return yieldOp.getOperation()->emitError()
           << "Currently do not support non-empty yield operations inside for "
              "loops. Run --scf-for-to-while before running --scf-to-calyx.";
  }

  if (auto whileOp = dyn_cast<scf::WhileOp>(yieldOp->getParentOp())) {
    ScfWhileOp whileOpInterface(whileOp);

    auto assignGroup =
        getState<ComponentLoweringState>().buildWhileLoopIterArgAssignments(
            rewriter, whileOpInterface,
            getState<ComponentLoweringState>().getComponentOp(),
            getState<ComponentLoweringState>().getUniqueName(whileOp) +
                "_latch",
            yieldOp->getOpOperands());
    getState<ComponentLoweringState>().setWhileLoopLatchGroup(whileOpInterface,
                                                              assignGroup);
    return success();
  }

  if (auto ifOp = dyn_cast<scf::IfOp>(yieldOp->getParentOp())) {
    if (ifOp.getResults().empty())
      return success();
    auto resultRegs = getState<ComponentLoweringState>().getResultRegs(ifOp);

    if (yieldOp->getParentRegion() == &ifOp.getThenRegion()) {
      auto thenGroup = getState<ComponentLoweringState>().getThenGroup(ifOp);
      for (auto op : enumerate(yieldOp.getOperands())) {
        auto resultReg =
            getState<ComponentLoweringState>().getResultRegs(ifOp, op.index());
        buildAssignmentsForRegisterWrite(
            rewriter, thenGroup,
            getState<ComponentLoweringState>().getComponentOp(), resultReg,
            op.value());
        getState<ComponentLoweringState>().registerEvaluatingGroup(
            ifOp.getResult(op.index()), thenGroup);
      }
    }

    if (!ifOp.getElseRegion().empty() &&
        (yieldOp->getParentRegion() == &ifOp.getElseRegion())) {
      auto elseGroup = getState<ComponentLoweringState>().getElseGroup(ifOp);
      for (auto op : enumerate(yieldOp.getOperands())) {
        auto resultReg =
            getState<ComponentLoweringState>().getResultRegs(ifOp, op.index());
        buildAssignmentsForRegisterWrite(
            rewriter, elseGroup,
            getState<ComponentLoweringState>().getComponentOp(), resultReg,
            op.value());
        getState<ComponentLoweringState>().registerEvaluatingGroup(
            ifOp.getResult(op.index()), elseGroup);
      }
    }
  }
  return success();
}

LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     BranchOpInterface brOp) const {
  /// Branch argument passing group creation
  /// Branch operands are passed through registers. In BuildBasicBlockRegs we
  /// created registers for all branch arguments of each block. We now
  /// create groups for assigning values to these registers.
  Block *srcBlock = brOp->getBlock();
  for (auto succBlock : enumerate(brOp->getSuccessors())) {
    auto succOperands = brOp.getSuccessorOperands(succBlock.index());
    if (succOperands.empty())
      continue;
    // Create operand passing group
    std::string groupName = loweringState().blockName(srcBlock) + "_to_" +
                            loweringState().blockName(succBlock.value());
    auto groupOp = calyx::createGroup<calyx::GroupOp>(rewriter, getComponent(),
                                                      brOp.getLoc(), groupName);
    // Fetch block argument registers associated with the basic block
    auto dstBlockArgRegs =
        getState<ComponentLoweringState>().getBlockArgRegs(succBlock.value());
    // Create register assignment for each block argument
    for (auto arg : enumerate(succOperands.getForwardedOperands())) {
      auto reg = dstBlockArgRegs[arg.index()];
      calyx::buildAssignmentsForRegisterWrite(
          rewriter, groupOp,
          getState<ComponentLoweringState>().getComponentOp(), reg,
          arg.value());
    }
    /// Register the group as a block argument group, to be executed
    /// when entering the successor block from this block (srcBlock).
    getState<ComponentLoweringState>().addBlockArgGroup(
        srcBlock, succBlock.value(), groupOp);
  }
  return success();
}

/// For each return statement, we create a new group for assigning to the
/// previously created return value registers.
LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     ReturnOp retOp) const {
  if (retOp.getNumOperands() == 0)
    return success();

  std::string groupName =
      getState<ComponentLoweringState>().getUniqueName("ret_assign");
  auto groupOp = calyx::createGroup<calyx::GroupOp>(rewriter, getComponent(),
                                                    retOp.getLoc(), groupName);
  for (auto op : enumerate(retOp.getOperands())) {
    auto reg = getState<ComponentLoweringState>().getReturnReg(op.index());
    calyx::buildAssignmentsForRegisterWrite(
        rewriter, groupOp, getState<ComponentLoweringState>().getComponentOp(),
        reg, op.value());
  }
  /// Schedule group for execution for when executing the return op block.
  getState<ComponentLoweringState>().addBlockScheduleable(retOp->getBlock(),
                                                          groupOp);
  return success();
}

LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     arith::ConstantOp constOp) const {
  if (isa<IntegerType>(constOp.getType())) {
    /// Move constant operations to the compOp body as hw::ConstantOp's.
    APInt value;
    calyx::matchConstantOp(constOp, value);
    auto hwConstOp =
        rewriter.replaceOpWithNewOp<hw::ConstantOp>(constOp, value);
    hwConstOp->moveAfter(getComponent().getBodyBlock(),
                         getComponent().getBodyBlock()->begin());
  } else {
    auto calyxConstOp = rewriter.replaceOpWithNewOp<calyx::ConstantOp>(
        constOp, constOp.getType(), constOp.getValueAttr());
    calyxConstOp->moveAfter(getComponent().getBodyBlock(),
                            getComponent().getBodyBlock()->begin());
  }

  return success();
}

LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     AddIOp op) const {
  return buildLibraryOp<calyx::CombGroupOp, calyx::AddLibOp>(rewriter, op);
}
LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     SubIOp op) const {
  return buildLibraryOp<calyx::CombGroupOp, calyx::SubLibOp>(rewriter, op);
}
LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     ShRUIOp op) const {
  return buildLibraryOp<calyx::CombGroupOp, calyx::RshLibOp>(rewriter, op);
}
LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     ShRSIOp op) const {
  return buildLibraryOp<calyx::CombGroupOp, calyx::SrshLibOp>(rewriter, op);
}
LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     ShLIOp op) const {
  return buildLibraryOp<calyx::CombGroupOp, calyx::LshLibOp>(rewriter, op);
}
LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     AndIOp op) const {
  return buildLibraryOp<calyx::CombGroupOp, calyx::AndLibOp>(rewriter, op);
}
LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     OrIOp op) const {
  return buildLibraryOp<calyx::CombGroupOp, calyx::OrLibOp>(rewriter, op);
}
LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     XOrIOp op) const {
  return buildLibraryOp<calyx::CombGroupOp, calyx::XorLibOp>(rewriter, op);
}
LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     SelectOp op) const {
  return buildLibraryOp<calyx::CombGroupOp, calyx::MuxLibOp>(rewriter, op);
}

LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     CmpIOp op) const {
  auto isPipeLibOp = [](Value val) -> bool {
    if (Operation *defOp = val.getDefiningOp()) {
      return isa<calyx::MultPipeLibOp, calyx::DivUPipeLibOp,
                 calyx::DivSPipeLibOp, calyx::RemUPipeLibOp,
                 calyx::RemSPipeLibOp>(defOp);
    }
    return false;
  };

  switch (op.getPredicate()) {
  case CmpIPredicate::eq: {
    StringRef opName = op.getOperationName().split(".").second;
    Type width = op.getResult().getType();
    bool isSequential = isPipeLibOp(op.getLhs()) || isPipeLibOp(op.getRhs());
    if (isSequential) {
      auto condReg = createRegister(
          op.getLoc(), rewriter, getComponent(), width.getIntOrFloatBitWidth(),
          getState<ComponentLoweringState>().getUniqueName(opName));

      for (auto *user : op->getUsers()) {
        if (auto ifOp = dyn_cast<scf::IfOp>(user))
          getState<ComponentLoweringState>().setCondReg(ifOp, condReg);
      }

      calyx::RegisterOp pipeResReg;
      if (isPipeLibOp(op.getLhs()))
        pipeResReg = getState<ComponentLoweringState>().getPipeResReg(
            op.getLhs().getDefiningOp());
      else
        pipeResReg = getState<ComponentLoweringState>().getPipeResReg(
            op.getRhs().getDefiningOp());

      return buildLibraryOp<calyx::GroupOp, calyx::EqLibOp>(
          rewriter, op, pipeResReg, condReg);
    }
    return buildLibraryOp<calyx::CombGroupOp, calyx::EqLibOp>(rewriter, op);
  }
  case CmpIPredicate::ne:
    return buildLibraryOp<calyx::CombGroupOp, calyx::NeqLibOp>(rewriter, op);
  case CmpIPredicate::uge:
    return buildLibraryOp<calyx::CombGroupOp, calyx::GeLibOp>(rewriter, op);
  case CmpIPredicate::ult:
    return buildLibraryOp<calyx::CombGroupOp, calyx::LtLibOp>(rewriter, op);
  case CmpIPredicate::ugt:
    return buildLibraryOp<calyx::CombGroupOp, calyx::GtLibOp>(rewriter, op);
  case CmpIPredicate::ule:
    return buildLibraryOp<calyx::CombGroupOp, calyx::LeLibOp>(rewriter, op);
  case CmpIPredicate::sge:
    return buildLibraryOp<calyx::CombGroupOp, calyx::SgeLibOp>(rewriter, op);
  case CmpIPredicate::slt:
    return buildLibraryOp<calyx::CombGroupOp, calyx::SltLibOp>(rewriter, op);
  case CmpIPredicate::sgt:
    return buildLibraryOp<calyx::CombGroupOp, calyx::SgtLibOp>(rewriter, op);
  case CmpIPredicate::sle:
    return buildLibraryOp<calyx::CombGroupOp, calyx::SleLibOp>(rewriter, op);
  }
  llvm_unreachable("unsupported comparison predicate");
}
LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     TruncIOp op) const {
  return buildLibraryOp<calyx::CombGroupOp, calyx::SliceLibOp>(
      rewriter, op, {op.getOperand().getType()}, {op.getType()});
}
LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     ExtUIOp op) const {
  return buildLibraryOp<calyx::CombGroupOp, calyx::PadLibOp>(
      rewriter, op, {op.getOperand().getType()}, {op.getType()});
}

LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     ExtSIOp op) const {
  return buildLibraryOp<calyx::CombGroupOp, calyx::ExtSILibOp>(
      rewriter, op, {op.getOperand().getType()}, {op.getType()});
}

LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     IndexCastOp op) const {
  Type sourceType = calyx::convIndexType(rewriter, op.getOperand().getType());
  Type targetType = calyx::convIndexType(rewriter, op.getResult().getType());
  unsigned targetBits = targetType.getIntOrFloatBitWidth();
  unsigned sourceBits = sourceType.getIntOrFloatBitWidth();
  LogicalResult res = success();

  if (targetBits == sourceBits) {
    /// Drop the index cast and replace uses of the target value with the source
    /// value.
    op.getResult().replaceAllUsesWith(op.getOperand());
  } else {
    /// pad/slice the source operand.
    if (sourceBits > targetBits)
      res = buildLibraryOp<calyx::CombGroupOp, calyx::SliceLibOp>(
          rewriter, op, {sourceType}, {targetType});
    else
      res = buildLibraryOp<calyx::CombGroupOp, calyx::PadLibOp>(
          rewriter, op, {sourceType}, {targetType});
  }
  rewriter.eraseOp(op);
  return res;
}

LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     scf::WhileOp whileOp) const {
  // Only need to add the whileOp to the BlockSchedulables scheduler interface.
  // Everything else was handled in the `BuildWhileGroups` pattern.
  ScfWhileOp scfWhileOp(whileOp);
  getState<ComponentLoweringState>().addBlockScheduleable(
      whileOp.getOperation()->getBlock(), WhileScheduleable{scfWhileOp});
  return success();
}

LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     scf::ForOp forOp) const {
  // Only need to add the forOp to the BlockSchedulables scheduler interface.
  // Everything else was handled in the `BuildForGroups` pattern.
  ScfForOp scfForOp(forOp);
  // If we cannot compute the trip count of the for loop, then we should
  // emit an error saying to use --scf-for-to-while
  std::optional<uint64_t> bound = scfForOp.getBound();
  if (!bound.has_value()) {
    return scfForOp.getOperation()->emitError()
           << "Loop bound not statically known. Should "
              "transform into while loop using `--scf-for-to-while` before "
              "running --lower-scf-to-calyx.";
  }
  getState<ComponentLoweringState>().addBlockScheduleable(
      forOp.getOperation()->getBlock(), ForScheduleable{
                                            scfForOp,
                                            bound.value(),
                                        });
  return success();
}

LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     scf::IfOp ifOp) const {
  getState<ComponentLoweringState>().addBlockScheduleable(
      ifOp.getOperation()->getBlock(), IfScheduleable{ifOp});
  return success();
}

LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     CallOp callOp) const {
  std::string instanceName = calyx::getInstanceName(callOp);
  calyx::InstanceOp instanceOp =
      getState<ComponentLoweringState>().getInstance(instanceName);
  SmallVector<Value, 4> outputPorts;
  auto portInfos = instanceOp.getReferencedComponent().getPortInfo();
  for (auto [idx, portInfo] : enumerate(portInfos)) {
    if (portInfo.direction == calyx::Direction::Output)
      outputPorts.push_back(instanceOp.getResult(idx));
  }

  // Replacing a CallOp results in the out port of the instance.
  for (auto [idx, result] : llvm::enumerate(callOp.getResults()))
    rewriter.replaceAllUsesWith(result, outputPorts[idx]);

  // CallScheduleanle requires an instance, while CallOp can be used to get the
  // input ports.
  getState<ComponentLoweringState>().addBlockScheduleable(
      callOp.getOperation()->getBlock(), CallScheduleable{instanceOp, callOp});
  return success();
}

/// Inlines Calyx ExecuteRegionOp operations within their parent blocks.
/// An execution region op (ERO) is inlined by:
///  i  : add a sink basic block for all yield operations inside the
///       ERO to jump to
///  ii : Rewrite scf.yield calls inside the ERO to branch to the sink block
///  iii: inline the ERO region
/// TODO(#1850) evaluate the usefulness of this lowering pattern.
class InlineExecuteRegionOpPattern
    : public OpRewritePattern<scf::ExecuteRegionOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ExecuteRegionOp execOp,
                                PatternRewriter &rewriter) const override {
    /// Determine type of "yield" operations inside the ERO.
    TypeRange yieldTypes = execOp.getResultTypes();

    /// Create sink basic block and rewrite uses of yield results to sink block
    /// arguments.
    rewriter.setInsertionPointAfter(execOp);
    auto *sinkBlock = rewriter.splitBlock(
        execOp->getBlock(),
        execOp.getOperation()->getIterator()->getNextNode()->getIterator());
    sinkBlock->addArguments(
        yieldTypes,
        SmallVector<Location, 4>(yieldTypes.size(), rewriter.getUnknownLoc()));
    for (auto res : enumerate(execOp.getResults()))
      res.value().replaceAllUsesWith(sinkBlock->getArgument(res.index()));

    /// Rewrite yield calls as branches.
    for (auto yieldOp :
         make_early_inc_range(execOp.getRegion().getOps<scf::YieldOp>())) {
      rewriter.setInsertionPointAfter(yieldOp);
      rewriter.replaceOpWithNewOp<BranchOp>(yieldOp, sinkBlock,
                                            yieldOp.getOperands());
    }

    /// Inline the regionOp.
    auto *preBlock = execOp->getBlock();
    auto *execOpEntryBlock = &execOp.getRegion().front();
    auto *postBlock = execOp->getBlock()->splitBlock(execOp);
    rewriter.inlineRegionBefore(execOp.getRegion(), postBlock);
    rewriter.mergeBlocks(postBlock, preBlock);
    rewriter.eraseOp(execOp);

    /// Finally, erase the unused entry block of the execOp region.
    rewriter.mergeBlocks(execOpEntryBlock, preBlock);

    return success();
  }
};

class MemoryBanking : public calyx::FuncOpPartialLoweringPattern {
  using FuncOpPartialLoweringPattern::FuncOpPartialLoweringPattern;

  LogicalResult
  partiallyLowerFuncToComp(FuncOp funcOp,
                           PatternRewriter &rewriter) const override {
    LogicalResult res = success();

    auto moduleOp = funcOp->getParentOfType<ModuleOp>();
    auto topLevelFn = cast<FuncOp>(SymbolTable::lookupSymbolIn(
        moduleOp, loweringState().getTopLevelFunction()));

    // Calyx puts the constraint that all memories should be defined in the
    // top-level function
    if (funcOp != topLevelFn)
      return res;

    auto allMemRefDefinitions = collectAllMemRefDefns(funcOp);

    for (auto *defn : allMemRefDefinitions) {
      if (auto bankAttr =
              defn->template getAttrOfType<IntegerAttr>("calyx.num_banks")) {
        uint availableBanks = bankAttr.getInt();
        if (availableBanks == 1)
          continue;
        auto banks = allocateBanks(rewriter, defn, availableBanks);

        if (failed(replaceMemUseWithBanks(rewriter, defn, banks)))
          return failure();
      }
    }

    return res;
  }

private:
  LogicalResult insertNewOperandsToCallSites(PatternRewriter &rewriter,
                                             CallOp callOp,
                                             SmallVector<Value> &newOperands,
                                             uint pos) const {
    auto callee = SymbolTable::lookupNearestSymbolFrom<FuncOp>(
        callOp, callOp.getCalleeAttr());

    SmallVector<Value> callerOperands(callOp.getOperands());
    if (pos > callerOperands.size())
      return rewriter.notifyMatchFailure(
          callOp, "position cannot be greater than caller operands size");
    // Insert the new operands after the specified position
    callerOperands.insert(callerOperands.begin() + pos + 1, newOperands.begin(),
                          newOperands.end());
    callOp->setOperands(callerOperands);
    if (callOp.getNumOperands() != callee.getNumArguments())
      return rewriter.notifyMatchFailure(
          callOp, "Number of operands and block arguments should match after "
                  "insertion");

    return success();
  }

  // Insert `newArgs` to `pos` argument in `funcOp`
  LogicalResult insertNewArgsToFunc(PatternRewriter &rewriter, FuncOp funcOp,
                                    SmallVector<Value> &newArgs,
                                    uint pos) const {
    // Get the current argument types
    SmallVector<Type, 4> updatedCurFnArgTys(funcOp.getArgumentTypes());
    // Collect the types of the new arguments
    SmallVector<Type, 4> newArgTys;
    for (auto arg : newArgs)
      newArgTys.push_back(arg.getType());

    if (pos > updatedCurFnArgTys.size())
      return rewriter.notifyMatchFailure(
          funcOp,
          "insert position cannot be larger than funcOp's argument numbers");

    // Insert newArgTys after the specified position
    updatedCurFnArgTys.insert(updatedCurFnArgTys.begin() + pos + 1,
                              newArgTys.begin(), newArgTys.end());
    // Insert the new arguments into the function body at the specified position
    Block &entryBlock = funcOp.getFunctionBody().front();
    unsigned insertPos = pos + 1;
    if (pos > entryBlock.getNumArguments())
      return rewriter.notifyMatchFailure(
          funcOp, "insert position cannot be larger than function body block's "
                  "argument numbers");

    for (size_t i = 0; i < newArgTys.size(); ++i) {
      // The position increases by 1 for each inserted argument
      entryBlock.insertArgument(insertPos + i, newArgTys[i], funcOp.getLoc());
    }

    // Create and set the new FunctionType with updated argument types
    auto newFnTy = FunctionType::get(funcOp.getContext(), updatedCurFnArgTys,
                                     funcOp.getResultTypes());
    funcOp.setType(newFnTy);

    return success();
  }

  Value computeIntraBankingOffset(PatternRewriter &rewriter, Value address,
                                  uint availableBanks) const {
    Value availBanksVal =
        rewriter
            .create<arith::ConstantOp>(rewriter.getUnknownLoc(),
                                       rewriter.getIndexAttr(availableBanks))
            .getResult();
    Value offset = rewriter
                       .create<arith::DivUIOp>(rewriter.getUnknownLoc(),
                                               address, availBanksVal)
                       .getResult();
    return offset;
  }

  void removeMemDefiningOp(Operation *memDefnOp) const {
    if (auto getGlobalOp = dyn_cast<memref::GetGlobalOp>(memDefnOp)) {
      auto *symbolTableOp =
          getGlobalOp->getParentWithTrait<mlir::OpTrait::SymbolTable>();
      auto globalOp =
          dyn_cast_or_null<memref::GlobalOp>(SymbolTable::lookupSymbolIn(
              symbolTableOp, getGlobalOp.getNameAttr()));
      getGlobalOp->remove();
      globalOp->remove();
    } else
      memDefnOp->remove();
  }

  // Replace the use of the BlockArgument at `argPos` with new BlockArguments
  LogicalResult replaceBlockArgWithNewArgs(PatternRewriter &rewriter,
                                           FuncOp funcOp, uint argPos,
                                           uint beginIdx, uint endIdx) const {
    IRMapping replaceMapping;

    auto oldArg = funcOp.getArgument(argPos);
    SmallVector<Value> replaceArgs(funcOp.getArguments().begin() + beginIdx,
                                   funcOp.getArguments().begin() + endIdx + 1);

    for (auto &use : llvm::make_early_inc_range(oldArg.getUses())) {
      Operation *op = use.getOwner();

      LogicalResult result =
          TypeSwitch<Operation *, LogicalResult>(op)
              .Case<memref::LoadOp, memref::StoreOp>([&](auto memUseOp) {
                if (!isUniDimensional(memUseOp.getMemRefType()))
                  return rewriter.notifyMatchFailure(
                      memUseOp, "all memories must be flattened before the "
                                "scf-to-calyx pass");

                // Replace the memory operation with banked mem ops that uses
                // memory references
                return replaceMemOpWithBankedMemOps(rewriter, memUseOp,
                                                    replaceArgs);
              })
              .Default([](Operation *) {
                llvm_unreachable("Unhandled memory operation type");
                return failure();
              });

      if (failed(result))
        return result;
    }

    return success();
  }

  LogicalResult eraseOperandFromCallSite(CallOp callOp, Value memrefVal) const {
    auto argPos = llvm::find(callOp.getOperands(), memrefVal) -
                  callOp.getOperands().begin();
    auto *definingMemOp = callOp.getOperand(argPos).getDefiningOp();
    callOp->eraseOperand(argPos);
    removeMemDefiningOp(definingMemOp);

    return success();
  }

  LogicalResult eraseArgInFunc(FuncOp funcOp, uint argPos) const {
    SmallVector<Type, 4> updatedCurFnArgTys(funcOp.getArgumentTypes());
    updatedCurFnArgTys.erase(updatedCurFnArgTys.begin() + argPos);
    funcOp.getBlocks().front().eraseArgument(argPos);
    auto newFnType = FunctionType::get(funcOp.getContext(), updatedCurFnArgTys,
                                       funcOp.getFunctionType().getResults());
    funcOp.setFunctionType(newFnType);

    return success();
  }

  // Replace `memOp`'s uses with `banks`
  LogicalResult replaceMemUseWithBanks(PatternRewriter &rewriter,
                                       Operation *memOp,
                                       SmallVector<Operation *> &banks) const {
    SmallVector<Value> bankResults;
    for (auto *bank : banks)
      bankResults.push_back(bank->getResult(0));

    for (auto &use : memOp->getResult(0).getUses()) {
      Operation *userOp = use.getOwner();
      if (auto callOp = dyn_cast<CallOp>(userOp)) {
        FuncOp calleeFunc = SymbolTable::lookupNearestSymbolFrom<FuncOp>(
            callOp, callOp.getCalleeAttr());
        auto pos = llvm::find(callOp.getOperands(), memOp->getResult(0)) -
                   callOp.getOperands().begin();
        if (failed(insertNewArgsToFunc(rewriter, calleeFunc, bankResults, pos)))
          return failure();

        if (failed(insertNewOperandsToCallSites(rewriter, callOp, bankResults,
                                                pos)))
          return failure();

        if (failed(replaceBlockArgWithNewArgs(rewriter, calleeFunc, pos,
                                              pos + 1, pos + banks.size() + 1)))
          return failure();

        if (calleeFunc.getArgument(pos).use_empty()) {
          if (failed(eraseOperandFromCallSite(callOp, memOp->getResult(0))))
            return failure();

          if (failed(eraseArgInFunc(calleeFunc, pos)))
            return failure();
        }
      } else {
        LogicalResult result =
            TypeSwitch<Operation *, LogicalResult>(userOp)
                .Case<memref::LoadOp, memref::StoreOp>([&](auto memUseOp) {
                  if (!isUniDimensional(memUseOp.getMemRefType()))
                    return rewriter.notifyMatchFailure(
                        memUseOp, "all memories must be flattened before the "
                                  "scf-to-calyx pass");

                  // Replace the memory operation with banked mems
                  return replaceMemOpWithBankedMemOps(rewriter, memUseOp,
                                                      bankResults);
                })
                .Default([](Operation *) {
                  llvm_unreachable("Unhandled memory operation type");
                  return failure();
                });

        if (failed(result))
          return result;
      }
    }

    return success();
  }

  LogicalResult
  replaceMemOpWithBankedMemOps(PatternRewriter &rewriter, Operation *memOp,
                               SmallVector<Value> &bankResults) const {
    Location loc = memOp->getLoc();
    rewriter.setInsertionPoint(memOp);
    TypeSwitch<Operation *>(memOp)
        .Case<memref::LoadOp, memref::StoreOp>([&](auto memUseOp) {
          // All memory has to be uni-dimensiona
          Value index = memUseOp.getIndices().front();
          unsigned numBanks = bankResults.size();
          Value numBanksValue =
              rewriter.create<arith::ConstantIndexOp>(loc, numBanks);

          // Compute bank index and local index within the bank
          Value bankIndex =
              rewriter.create<arith::RemUIOp>(loc, index, numBanksValue);
          Value bankAddress =
              computeIntraBankingOffset(rewriter, index, numBanks);

          // Create switchOp to select the bank
          SmallVector<Type> resultTypes = {};
          if (isa<memref::LoadOp>(memUseOp)) {
            auto loadOp = cast<memref::LoadOp>(memUseOp);
            resultTypes = {loadOp.getType()};
          }

          SmallVector<int64_t> caseValues;
          for (unsigned i = 0; i < numBanks; ++i)
            caseValues.push_back(i);

          scf::IndexSwitchOp switchOp = rewriter.create<scf::IndexSwitchOp>(
              loc, resultTypes, bankIndex, caseValues, /*numRegions=*/numBanks);

          // Populate the case regions
          for (unsigned i = 0; i < numBanks; ++i) {
            Region &caseRegion = switchOp.getCaseRegions()[i];
            rewriter.setInsertionPointToStart(&caseRegion.emplaceBlock());

            Value bankMemRef = bankResults[i];
            if (isa<memref::LoadOp>(memUseOp)) {
              auto newLoadOp =
                  rewriter.create<memref::LoadOp>(loc, bankMemRef, bankAddress);
              // Yield the result of the new load operation
              rewriter.create<scf::YieldOp>(loc, newLoadOp.getResult());
            } else {
              auto storeOp = cast<memref::StoreOp>(memUseOp);
              rewriter.create<memref::StoreOp>(loc, storeOp.getValueToStore(),
                                               bankMemRef, bankAddress);
              rewriter.create<scf::YieldOp>(loc);
            }
          }

          Region &defaultRegion = switchOp.getDefaultRegion();
          assert(defaultRegion.empty() && "Default region should be empty");
          rewriter.setInsertionPointToStart(&defaultRegion.emplaceBlock());

          if (isa<memref::LoadOp>(memUseOp)) {
            auto loadOp = cast<memref::LoadOp>(memUseOp);
            Type loadType = loadOp.getType();
            TypedAttr zeroAttr =
                cast<TypedAttr>(rewriter.getZeroAttr(loadType));
            auto defaultValue =
                rewriter.create<arith::ConstantOp>(loc, zeroAttr);
            // Yield the default value zero
            rewriter.create<scf::YieldOp>(loc, defaultValue.getResult());

            // Replace the original loadOp's result with the switchOp's result
            loadOp.getResult().replaceAllUsesWith(switchOp.getResult(0));
          } else
            rewriter.create<scf::YieldOp>(loc);

          rewriter.eraseOp(memUseOp);
        })
        .Default([](Operation *) {
          llvm_unreachable("Unhandled memory operation type");
        });

    return success();
  }

  SmallVector<Operation *> collectAllMemRefDefns(FuncOp funcOp) const {
    SmallVector<Operation *> allMemRefDefinitions;
    funcOp.walk([&](Operation *op) {
      if (isa<memref::AllocOp, memref::AllocaOp, memref::GetGlobalOp>(op)) {
        allMemRefDefinitions.push_back(op);
      }
    });

    return allMemRefDefinitions;
  }

  mutable std::atomic<unsigned> globalCounter;
  SmallVector<Operation *> allocateBanks(PatternRewriter &rewriter,
                                         Operation *definingMemOp,
                                         uint availableBanks) const {
    auto ensureTypeValidForBanking = [&availableBanks](MemRefType memTy) {
      auto memShape = memTy.getShape();
      assert(circt::isUniDimensional(memTy) &&
             "all memories must be flattened before the scf-to-calyx pass");
      assert(memShape.front() % availableBanks == 0 &&
             "memory size must be divisible by the banking factor");
    };

    SmallVector<Operation *> banks;
    Location loc = definingMemOp->getParentOp()->getLoc();
    rewriter.setInsertionPointAfter(definingMemOp);

    TypeSwitch<Operation *>(definingMemOp)
        .Case<memref::AllocOp>([&](memref::AllocOp allocOp) {
          auto memTy = cast<MemRefType>(allocOp.getMemref().getType());
          ensureTypeValidForBanking(memTy);

          uint bankSize = memTy.getShape().front() / availableBanks;
          for (uint bankCnt = 0; bankCnt < availableBanks; bankCnt++) {
            auto bankAllocOp = rewriter.create<memref::AllocOp>(
                loc,
                MemRefType::get(bankSize, memTy.getElementType(),
                                memTy.getLayout(), memTy.getMemorySpace()));
            banks.push_back(bankAllocOp);
          }
        })
        .Case<memref::GetGlobalOp>([&](memref::GetGlobalOp getGlobalOp) {
          auto memTy = cast<MemRefType>(getGlobalOp.getType());
          ensureTypeValidForBanking(memTy);

          OpBuilder::InsertPoint globalOpsInsertPt, getGlobalOpsInsertPt;
          uint bankSize = memTy.getShape().front() / availableBanks;
          for (uint bankCnt = 0; bankCnt < availableBanks; bankCnt++) {
            auto *symbolTableOp =
                getGlobalOp->getParentWithTrait<OpTrait::SymbolTable>();
            auto globalOp =
                dyn_cast_or_null<memref::GlobalOp>(SymbolTable::lookupSymbolIn(
                    symbolTableOp, getGlobalOp.getNameAttr()));
            MemRefType globalOpTy = globalOp.getType();
            auto cstAttr = llvm::dyn_cast_or_null<DenseElementsAttr>(
                globalOp.getConstantInitValue());
            auto allAttrs = cstAttr.getValues<Attribute>();
            uint beginIdx = bankSize * bankCnt;
            uint endIdx = bankSize * (bankCnt + 1);
            SmallVector<Attribute, 8> extractedElements;
            for (uint i = 0; i < bankSize; i++) {
              uint idx = bankCnt + availableBanks * i;
              extractedElements.push_back(allAttrs[idx]);
            }

            if (bankCnt == 0) {
              rewriter.setInsertionPointAfter(globalOp);
              globalOpsInsertPt = rewriter.saveInsertionPoint();
              rewriter.setInsertionPointAfter(getGlobalOp);
              getGlobalOpsInsertPt = rewriter.saveInsertionPoint();
            }

            // Prepare relevant information to create a new `GlobalOp`
            auto newMemRefTy =
                MemRefType::get(SmallVector<int64_t>{endIdx - beginIdx},
                                globalOpTy.getElementType());
            auto newTypeAttr = TypeAttr::get(newMemRefTy);
            std::string newNameStr = llvm::formatv(
                "{0}_{1}x{2}_{3}_{4}", globalOp.getConstantAttrName(),
                endIdx - beginIdx, globalOpTy.getElementType(), bankCnt,
                globalCounter++);
            RankedTensorType tensorType = RankedTensorType::get(
                {static_cast<int64_t>(extractedElements.size())},
                globalOpTy.getElementType());
            auto newInitValue =
                DenseElementsAttr::get(tensorType, extractedElements);

            // Create a new `GlobalOp`
            rewriter.restoreInsertionPoint(globalOpsInsertPt);
            auto newGlobalOp = rewriter.create<memref::GlobalOp>(
                loc, rewriter.getStringAttr(newNameStr),
                globalOp.getSymVisibilityAttr(), newTypeAttr, newInitValue,
                globalOp.getConstantAttr(), globalOp.getAlignmentAttr());
            rewriter.setInsertionPointAfter(newGlobalOp);
            globalOpsInsertPt = rewriter.saveInsertionPoint();

            // Create a new `GetGlobalOp` using the above created new `GlobalOp`
            rewriter.restoreInsertionPoint(getGlobalOpsInsertPt);
            auto newGetGlobalOp = rewriter.create<memref::GetGlobalOp>(
                loc, newMemRefTy, newGlobalOp.getName());
            rewriter.setInsertionPointAfter(newGetGlobalOp);
            getGlobalOpsInsertPt = rewriter.saveInsertionPoint();

            banks.push_back(newGetGlobalOp);
          }
        })
        .Default([](Operation *) {
          llvm_unreachable("Unhandled memory operation type");
        });
    return banks;
  }
};

/// Creates a new Calyx component for each FuncOp in the program.
struct FuncOpConversion : public calyx::FuncOpPartialLoweringPattern {
  using FuncOpPartialLoweringPattern::FuncOpPartialLoweringPattern;

  LogicalResult
  partiallyLowerFuncToComp(FuncOp funcOp,
                           PatternRewriter &rewriter) const override {
    /// Maintain a mapping between funcOp input arguments and the port index
    /// which the argument will eventually map to.
    DenseMap<Value, unsigned> funcOpArgRewrites;

    /// Maintain a mapping between funcOp output indexes and the component
    /// output port index which the return value will eventually map to.
    DenseMap<unsigned, unsigned> funcOpResultMapping;

    /// Maintain a mapping between an external memory argument (identified by a
    /// memref) and eventual component input- and output port indices that will
    /// map to the memory ports. The pair denotes the start index of the memory
    /// ports in the in- and output ports of the component. Ports are expected
    /// to be ordered in the same manner as they are added by
    /// calyx::appendPortsForExternalMemref.
    DenseMap<Value, std::pair<unsigned, unsigned>> extMemoryCompPortIndices;

    /// Create I/O ports. Maintain separate in/out port vectors to determine
    /// which port index each function argument will eventually map to.
    SmallVector<calyx::PortInfo> inPorts, outPorts;
    FunctionType funcType = funcOp.getFunctionType();
    for (auto arg : enumerate(funcOp.getArguments())) {
      if (!isa<MemRefType>(arg.value().getType())) {
        /// Single-port arguments
        std::string inName;
        if (auto portNameAttr = funcOp.getArgAttrOfType<StringAttr>(
                arg.index(), scfToCalyx::sPortNameAttr))
          inName = portNameAttr.str();
        else
          inName = "in" + std::to_string(arg.index());
        funcOpArgRewrites[arg.value()] = inPorts.size();
        inPorts.push_back(calyx::PortInfo{
            rewriter.getStringAttr(inName),
            calyx::convIndexType(rewriter, arg.value().getType()),
            calyx::Direction::Input,
            DictionaryAttr::get(rewriter.getContext(), {})});
      }
    }
    for (auto res : enumerate(funcType.getResults())) {
      std::string resName;
      if (auto portNameAttr = funcOp.getResultAttrOfType<StringAttr>(
              res.index(), scfToCalyx::sPortNameAttr))
        resName = portNameAttr.str();
      else
        resName = "out" + std::to_string(res.index());
      funcOpResultMapping[res.index()] = outPorts.size();
      outPorts.push_back(calyx::PortInfo{
          rewriter.getStringAttr(resName),
          calyx::convIndexType(rewriter, res.value()), calyx::Direction::Output,
          DictionaryAttr::get(rewriter.getContext(), {})});
    }

    /// We've now recorded all necessary indices. Merge in- and output ports
    /// and add the required mandatory component ports.
    auto ports = inPorts;
    llvm::append_range(ports, outPorts);
    calyx::addMandatoryComponentPorts(rewriter, ports);

    /// Create a calyx::ComponentOp corresponding to the to-be-lowered function.
    auto compOp = rewriter.create<calyx::ComponentOp>(
        funcOp.getLoc(), rewriter.getStringAttr(funcOp.getSymName()), ports);

    std::string funcName = "func_" + funcOp.getSymName().str();
    rewriter.modifyOpInPlace(funcOp, [&]() { funcOp.setSymName(funcName); });

    /// Mark this component as the toplevel if it's the top-level function of
    /// the module.
    if (compOp.getName() == loweringState().getTopLevelFunction())
      compOp->setAttr("toplevel", rewriter.getUnitAttr());

    /// Store the function-to-component mapping.
    functionMapping[funcOp] = compOp;
    auto *compState = loweringState().getState<ComponentLoweringState>(compOp);
    compState->setFuncOpResultMapping(funcOpResultMapping);

    unsigned extMemCounter = 0;
    for (auto arg : enumerate(funcOp.getArguments())) {
      if (isa<MemRefType>(arg.value().getType())) {
        std::string memName =
            llvm::join_items("_", "arg_mem", std::to_string(extMemCounter++));

        rewriter.setInsertionPointToStart(compOp.getBodyBlock());
        MemRefType memtype = cast<MemRefType>(arg.value().getType());
        SmallVector<int64_t> addrSizes;
        SmallVector<int64_t> sizes;
        for (int64_t dim : memtype.getShape()) {
          sizes.push_back(dim);
          addrSizes.push_back(calyx::handleZeroWidth(dim));
        }
        if (sizes.empty() && addrSizes.empty()) {
          sizes.push_back(1);
          addrSizes.push_back(1);
        }
        auto memOp = rewriter.create<calyx::MemoryOp>(
            funcOp.getLoc(), memName,
            memtype.getElementType(), sizes, addrSizes);
        // we don't set the memory to "external", which implies it's a reference

        compState->registerMemoryInterface(arg.value(),
                                           calyx::MemoryInterface(memOp));
      }
    }

    /// Rewrite funcOp SSA argument values to the CompOp arguments.
    for (auto &mapping : funcOpArgRewrites)
      mapping.getFirst().replaceAllUsesWith(
          compOp.getArgument(mapping.getSecond()));

    return success();
  }
};

/// In BuildWhileGroups, a register is created for each iteration argumenet of
/// the while op. These registers are then written to on the while op
/// terminating yield operation alongside before executing the whileOp in the
/// schedule, to set the initial values of the argument registers.
class BuildWhileGroups : public calyx::FuncOpPartialLoweringPattern {
  using FuncOpPartialLoweringPattern::FuncOpPartialLoweringPattern;

  LogicalResult
  partiallyLowerFuncToComp(FuncOp funcOp,
                           PatternRewriter &rewriter) const override {
    LogicalResult res = success();
    funcOp.walk([&](Operation *op) {
      // Only work on ops that support the ScfWhileOp.
      if (!isa<scf::WhileOp>(op))
        return WalkResult::advance();

      auto scfWhileOp = cast<scf::WhileOp>(op);
      ScfWhileOp whileOp(scfWhileOp);

      getState<ComponentLoweringState>().setUniqueName(whileOp.getOperation(),
                                                       "while");

      /// Check for do-while loops.
      /// TODO(mortbopet) can we support these? for now, do not support loops
      /// where iterargs are changed in the 'before' region. scf.WhileOp also
      /// has support for different types of iter_args and return args which we
      /// also do not support; iter_args and while return values are placed in
      /// the same registers.
      for (auto barg :
           enumerate(scfWhileOp.getBefore().front().getArguments())) {
        auto condOp = scfWhileOp.getConditionOp().getArgs()[barg.index()];
        if (barg.value() != condOp) {
          res = whileOp.getOperation()->emitError()
                << loweringState().irName(barg.value())
                << " != " << loweringState().irName(condOp)
                << "do-while loops not supported; expected iter-args to "
                   "remain untransformed in the 'before' region of the "
                   "scf.while op.";
          return WalkResult::interrupt();
        }
      }

      /// Create iteration argument registers.
      /// The iteration argument registers will be referenced:
      /// - In the "before" part of the while loop, calculating the conditional,
      /// - In the "after" part of the while loop,
      /// - Outside the while loop, rewriting the while loop return values.
      for (auto arg : enumerate(whileOp.getBodyArgs())) {
        std::string name = getState<ComponentLoweringState>()
                               .getUniqueName(whileOp.getOperation())
                               .str() +
                           "_arg" + std::to_string(arg.index());
        auto reg =
            createRegister(arg.value().getLoc(), rewriter, getComponent(),
                           arg.value().getType().getIntOrFloatBitWidth(), name);
        getState<ComponentLoweringState>().addWhileLoopIterReg(whileOp, reg,
                                                               arg.index());
        arg.value().replaceAllUsesWith(reg.getOut());

        /// Also replace uses in the "before" region of the while loop
        whileOp.getConditionBlock()
            ->getArgument(arg.index())
            .replaceAllUsesWith(reg.getOut());
      }

      /// Create iter args initial value assignment group(s), one per register.
      SmallVector<calyx::GroupOp> initGroups;
      auto numOperands = whileOp.getOperation()->getNumOperands();
      for (size_t i = 0; i < numOperands; ++i) {
        auto initGroupOp =
            getState<ComponentLoweringState>().buildWhileLoopIterArgAssignments(
                rewriter, whileOp,
                getState<ComponentLoweringState>().getComponentOp(),
                getState<ComponentLoweringState>().getUniqueName(
                    whileOp.getOperation()) +
                    "_init_" + std::to_string(i),
                whileOp.getOperation()->getOpOperand(i));
        initGroups.push_back(initGroupOp);
      }

      getState<ComponentLoweringState>().setWhileLoopInitGroups(whileOp,
                                                                initGroups);

      return WalkResult::advance();
    });
    return res;
  }
};

/// In BuildForGroups, a register is created for the iteration argument of
/// the for op. This register is then initialized to the lowerBound of the for
/// loop in a group that executes the for loop.
class BuildForGroups : public calyx::FuncOpPartialLoweringPattern {
  using FuncOpPartialLoweringPattern::FuncOpPartialLoweringPattern;

  LogicalResult
  partiallyLowerFuncToComp(FuncOp funcOp,
                           PatternRewriter &rewriter) const override {
    LogicalResult res = success();
    funcOp.walk([&](Operation *op) {
      // Only work on ops that support the ScfForOp.
      if (!isa<scf::ForOp>(op))
        return WalkResult::advance();

      auto scfForOp = cast<scf::ForOp>(op);
      ScfForOp forOp(scfForOp);

      getState<ComponentLoweringState>().setUniqueName(forOp.getOperation(),
                                                       "for");

      // Create a register for the InductionVar, and set that Register as the
      // only IterReg for the For Loop
      auto inductionVar = forOp.getOperation().getInductionVar();
      SmallVector<std::string, 3> inductionVarIdentifiers = {
          getState<ComponentLoweringState>()
              .getUniqueName(forOp.getOperation())
              .str(),
          "induction", "var"};
      std::string name = llvm::join(inductionVarIdentifiers, "_");
      auto reg =
          createRegister(inductionVar.getLoc(), rewriter, getComponent(),
                         inductionVar.getType().getIntOrFloatBitWidth(), name);
      getState<ComponentLoweringState>().addForLoopIterReg(forOp, reg, 0);
      inductionVar.replaceAllUsesWith(reg.getOut());

      // Create InitGroup that sets the InductionVar to LowerBound
      calyx::ComponentOp componentOp =
          getState<ComponentLoweringState>().getComponentOp();
      SmallVector<calyx::GroupOp> initGroups;
      SmallVector<std::string, 4> groupIdentifiers = {
          "init",
          getState<ComponentLoweringState>()
              .getUniqueName(forOp.getOperation())
              .str(),
          "induction", "var"};
      std::string groupName = llvm::join(groupIdentifiers, "_");
      auto groupOp = calyx::createGroup<calyx::GroupOp>(
          rewriter, componentOp, forOp.getLoc(), groupName);
      buildAssignmentsForRegisterWrite(rewriter, groupOp, componentOp, reg,
                                       forOp.getOperation().getLowerBound());
      initGroups.push_back(groupOp);
      getState<ComponentLoweringState>().setForLoopInitGroups(forOp,
                                                              initGroups);

      return WalkResult::advance();
    });
    return res;
  }
};

class BuildIfGroups : public calyx::FuncOpPartialLoweringPattern {
  using FuncOpPartialLoweringPattern::FuncOpPartialLoweringPattern;

  LogicalResult
  partiallyLowerFuncToComp(FuncOp funcOp,
                           PatternRewriter &rewriter) const override {
    LogicalResult res = success();
    funcOp.walk([&](Operation *op) {
      if (!isa<scf::IfOp>(op))
        return WalkResult::advance();

      auto scfIfOp = cast<scf::IfOp>(op);

      calyx::ComponentOp componentOp =
          getState<ComponentLoweringState>().getComponentOp();

      if (!scfIfOp.getResults().empty()) {
        std::string thenGroupName =
            getState<ComponentLoweringState>().getUniqueName("then_br");
        auto thenGroupOp = calyx::createGroup<calyx::GroupOp>(
            rewriter, componentOp, scfIfOp.getLoc(), thenGroupName);
        getState<ComponentLoweringState>().setThenGroup(scfIfOp, thenGroupOp);
      }

      if (!scfIfOp.getElseRegion().empty() && !scfIfOp.getResults().empty()) {
        std::string elseGroupName =
            getState<ComponentLoweringState>().getUniqueName("else_br");
        auto elseGroupOp = calyx::createGroup<calyx::GroupOp>(
            rewriter, componentOp, scfIfOp.getLoc(), elseGroupName);
        getState<ComponentLoweringState>().setElseGroup(scfIfOp, elseGroupOp);
      }

      for (auto ifOpRes : scfIfOp.getResults()) {
        auto reg = createRegister(
            scfIfOp.getLoc(), rewriter, getComponent(),
            ifOpRes.getType().getIntOrFloatBitWidth(),
            getState<ComponentLoweringState>().getUniqueName("if_res"));
        getState<ComponentLoweringState>().setResultRegs(
            scfIfOp, reg, ifOpRes.getResultNumber());
      }

      return WalkResult::advance();
    });
    return res;
  }
};

class BuildSwitchGroups : public calyx::FuncOpPartialLoweringPattern {
  using FuncOpPartialLoweringPattern::FuncOpPartialLoweringPattern;

  LogicalResult
  partiallyLowerFuncToComp(FuncOp funcOp,
                           PatternRewriter &rewriter) const override {
    LogicalResult res = success();
    funcOp.walk([&](Operation *op) {
      if (!isa<scf::IndexSwitchOp>(op))
        return WalkResult::advance();

      auto switchOp = cast<scf::IndexSwitchOp>(op);
      auto loc = switchOp.getLoc();

      Region &defaultRegion = switchOp.getDefaultRegion();
      Operation *yieldOp = defaultRegion.front().getTerminator();
      Value defaultResult = {};
      if (!yieldOp->getOperands().empty())
        defaultResult = yieldOp->getOperand(0);

      Value finalResult = defaultResult;
      scf::IfOp prevIfOp = nullptr;

      rewriter.setInsertionPointAfter(switchOp);
      for (size_t i = 0; i < switchOp.getCases().size(); i++) {
        auto caseValueInt = switchOp.getCases()[i];
        if (prevIfOp && !prevIfOp.getElseRegion().empty())
          rewriter.setInsertionPointToStart(&prevIfOp.getElseRegion().front());

        Value caseValue = rewriter.create<ConstantIndexOp>(loc, caseValueInt);
        Value cond = rewriter.create<CmpIOp>(
            loc, CmpIPredicate::eq, *switchOp.getODSOperands(0).begin(),
            caseValue);

        bool hasElseRegion =
            (i < switchOp.getCases().size() - 1 || defaultResult);
        auto ifOp = rewriter.create<scf::IfOp>(loc, switchOp.getResultTypes(),
                                               cond, hasElseRegion);

        Region &caseRegion = switchOp.getCaseRegions()[i];
        IRMapping mapping;
        Block &emptyThenBlock = ifOp.getThenRegion().front();
        emptyThenBlock.erase();
        caseRegion.cloneInto(&ifOp.getThenRegion(), mapping);

        if (i == switchOp.getCases().size() - 1 && defaultResult) {
          rewriter.setInsertionPointToEnd(&ifOp.getElseRegion().front());
          rewriter.create<scf::YieldOp>(loc, defaultResult);
        }

        if (prevIfOp && !prevIfOp.getElseRegion().empty()) {
          rewriter.setInsertionPointToEnd(&prevIfOp.getElseRegion().front());
          if (!ifOp.getResults().empty())
            rewriter.create<scf::YieldOp>(loc, ifOp.getResult(0));
        }

        if (i == 0 && !ifOp.getResults().empty())
          finalResult = ifOp.getResult(0);

        prevIfOp = ifOp;
      }

      if (switchOp.getNumResults() == 0)
        rewriter.eraseOp(switchOp);
      else
        rewriter.replaceOp(switchOp, finalResult);

      return WalkResult::advance();
    });
    funcOp.dump();
    return res;
  }
};

/// Builds a control schedule by traversing the CFG of the function and
/// associating this with the previously created groups.
/// For simplicity, the generated control flow is expanded for all possible
/// paths in the input DAG. This elaborated control flow is later reduced in
/// the runControlFlowSimplification passes.
class BuildControl : public calyx::FuncOpPartialLoweringPattern {
  using FuncOpPartialLoweringPattern::FuncOpPartialLoweringPattern;

  LogicalResult
  partiallyLowerFuncToComp(FuncOp funcOp,
                           PatternRewriter &rewriter) const override {
    auto *entryBlock = &funcOp.getBlocks().front();
    rewriter.setInsertionPointToStart(
        getComponent().getControlOp().getBodyBlock());
    auto topLevelSeqOp = rewriter.create<calyx::SeqOp>(funcOp.getLoc());
    DenseSet<Block *> path;
    return buildCFGControl(path, rewriter, topLevelSeqOp.getBodyBlock(),
                           nullptr, entryBlock);
  }

private:
  /// Sequentially schedules the groups that registered themselves with
  /// 'block'.
  LogicalResult scheduleBasicBlock(PatternRewriter &rewriter,
                                   const DenseSet<Block *> &path,
                                   mlir::Block *parentCtrlBlock,
                                   mlir::Block *block) const {
    auto compBlockScheduleables =
        getState<ComponentLoweringState>().getBlockScheduleables(block);
    auto loc = block->front().getLoc();

    if (compBlockScheduleables.size() > 1) {
      auto seqOp = rewriter.create<calyx::SeqOp>(loc);
      parentCtrlBlock = seqOp.getBodyBlock();
    }

    for (auto &group : compBlockScheduleables) {
      rewriter.setInsertionPointToEnd(parentCtrlBlock);
      if (auto *groupPtr = std::get_if<calyx::GroupOp>(&group); groupPtr) {
        rewriter.create<calyx::EnableOp>(groupPtr->getLoc(),
                                         groupPtr->getSymName());
      } else if (auto *whileSchedPtr = std::get_if<WhileScheduleable>(&group);
                 whileSchedPtr) {
        auto &whileOp = whileSchedPtr->whileOp;

        auto whileCtrlOp = buildWhileCtrlOp(
            whileOp,
            getState<ComponentLoweringState>().getWhileLoopInitGroups(whileOp),
            rewriter);
        rewriter.setInsertionPointToEnd(whileCtrlOp.getBodyBlock());
        auto whileBodyOp =
            rewriter.create<calyx::SeqOp>(whileOp.getOperation()->getLoc());
        auto *whileBodyOpBlock = whileBodyOp.getBodyBlock();

        /// Only schedule the 'after' block. The 'before' block is
        /// implicitly scheduled when evaluating the while condition.
        LogicalResult res = buildCFGControl(path, rewriter, whileBodyOpBlock,
                                            block, whileOp.getBodyBlock());

        // Insert loop-latch at the end of the while group
        rewriter.setInsertionPointToEnd(whileBodyOpBlock);
        calyx::GroupOp whileLatchGroup =
            getState<ComponentLoweringState>().getWhileLoopLatchGroup(whileOp);
        rewriter.create<calyx::EnableOp>(whileLatchGroup.getLoc(),
                                         whileLatchGroup.getName());

        if (res.failed())
          return res;
      } else if (auto *forSchedPtr = std::get_if<ForScheduleable>(&group);
                 forSchedPtr) {
        auto forOp = forSchedPtr->forOp;

        auto forCtrlOp = buildForCtrlOp(
            forOp,
            getState<ComponentLoweringState>().getForLoopInitGroups(forOp),
            forSchedPtr->bound, rewriter);
        rewriter.setInsertionPointToEnd(forCtrlOp.getBodyBlock());
        auto forBodyOp =
            rewriter.create<calyx::SeqOp>(forOp.getOperation()->getLoc());
        auto *forBodyOpBlock = forBodyOp.getBodyBlock();

        // Schedule the body of the for loop.
        LogicalResult res = buildCFGControl(path, rewriter, forBodyOpBlock,
                                            block, forOp.getBodyBlock());

        // Insert loop-latch at the end of the while group.
        rewriter.setInsertionPointToEnd(forBodyOpBlock);
        calyx::GroupOp forLatchGroup =
            getState<ComponentLoweringState>().getForLoopLatchGroup(forOp);
        rewriter.create<calyx::EnableOp>(forLatchGroup.getLoc(),
                                         forLatchGroup.getName());
        if (res.failed())
          return res;
      } else if (auto *ifSchedPtr = std::get_if<IfScheduleable>(&group);
                 ifSchedPtr) {
        auto ifOp = ifSchedPtr->ifOp;

        Location loc = ifOp->getLoc();

        auto cond = ifOp.getCondition();

        FlatSymbolRefAttr symbolAttr = nullptr;
        auto condReg = getState<ComponentLoweringState>().getCondReg(ifOp);
        if (!condReg) {
          auto condGroup = getState<ComponentLoweringState>()
                               .getEvaluatingGroup<calyx::CombGroupOp>(cond);

          symbolAttr = FlatSymbolRefAttr::get(
              StringAttr::get(getContext(), condGroup.getSymName()));
        }

        bool initElse = !ifOp.getElseRegion().empty();
        auto ifCtrlOp = rewriter.create<calyx::IfOp>(
            loc, cond, symbolAttr, /*initializeElseBody=*/initElse);

        rewriter.setInsertionPointToEnd(ifCtrlOp.getBodyBlock());

        auto thenSeqOp =
            rewriter.create<calyx::SeqOp>(ifOp.getThenRegion().getLoc());
        auto *thenSeqOpBlock = thenSeqOp.getBodyBlock();

        auto *thenBlock = &ifOp.getThenRegion().front();
        LogicalResult res = buildCFGControl(path, rewriter, thenSeqOpBlock,
                                            /*preBlock=*/block, thenBlock);
        if (res.failed())
          return res;

        if (!ifOp.getResults().empty()) {
          rewriter.setInsertionPointToEnd(thenSeqOpBlock);
          calyx::GroupOp thenGroup =
              getState<ComponentLoweringState>().getThenGroup(ifOp);
          rewriter.create<calyx::EnableOp>(thenGroup.getLoc(),
                                           thenGroup.getName());
        }

        if (!ifOp.getElseRegion().empty()) {
          rewriter.setInsertionPointToEnd(ifCtrlOp.getElseBody());

          auto elseSeqOp =
              rewriter.create<calyx::SeqOp>(ifOp.getElseRegion().getLoc());
          auto *elseSeqOpBlock = elseSeqOp.getBodyBlock();

          auto *elseBlock = &ifOp.getElseRegion().front();
          res = buildCFGControl(path, rewriter, elseSeqOpBlock,
                                /*preBlock=*/block, elseBlock);
          if (res.failed())
            return res;

          if (!ifOp.getResults().empty()) {
            rewriter.setInsertionPointToEnd(elseSeqOpBlock);
            calyx::GroupOp elseGroup =
                getState<ComponentLoweringState>().getElseGroup(ifOp);
            rewriter.create<calyx::EnableOp>(elseGroup.getLoc(),
                                             elseGroup.getName());
          }
        }

        getState<ComponentLoweringState>().getComponentOp().dump();
      } else if (auto *callSchedPtr = std::get_if<CallScheduleable>(&group)) {
        auto instanceOp = callSchedPtr->instanceOp;
        auto callBody = rewriter.create<calyx::SeqOp>(instanceOp.getLoc());
        rewriter.setInsertionPointToStart(callBody.getBodyBlock());
        
        auto callee = callSchedPtr->callOp.getCallee();
        auto *calleeOp = SymbolTable::lookupNearestSymbolFrom(
            callSchedPtr->callOp.getOperation()->getParentOp(),
            StringAttr::get(rewriter.getContext(), "func_" + callee.str()));
        FuncOp calleeFunc = dyn_cast_or_null<FuncOp>(calleeOp);

        auto instanceOpComp =
            llvm::cast<calyx::ComponentOp>(instanceOp.getReferencedComponent());
        auto *instanceOpLoweringState = loweringState().getState(instanceOpComp);
        
        SmallVector<Value, 4> instancePorts;
        SmallVector<Value, 4> inputPorts;
        SmallVector<Attribute, 4> refCells;
        for (auto operandEnum : enumerate(callSchedPtr->callOp.getOperands())) {
          auto operand = operandEnum.value();
          auto index = operandEnum.index();
          if (isa<MemRefType>(operand.getType())) {
            auto memOpName = getState<ComponentLoweringState>()
                                 .getMemoryInterface(operand)
                                 .memName();
            auto memOpNameAttr =
                SymbolRefAttr::get(rewriter.getContext(), memOpName);
            Value argI = calleeFunc.getArgument(index);

            if (isa<MemRefType>(argI.getType())) {
              NamedAttrList namedAttrList;
              namedAttrList.append(
                rewriter.getStringAttr(instanceOpLoweringState->getMemoryInterface(argI).memName()), memOpNameAttr);
              refCells.push_back(DictionaryAttr::get(rewriter.getContext(), namedAttrList));
            }
          } else {
            inputPorts.push_back(operand);
          }
        }
        llvm::copy(instanceOp.getResults().take_front(inputPorts.size()),
                   std::back_inserter(instancePorts));
 
        ArrayAttr refCellsAttr = ArrayAttr::get(rewriter.getContext(), refCells);
        rewriter.create<calyx::InvokeOp>(
            instanceOp.getLoc(), instanceOp.getSymName(), instancePorts,
            inputPorts, refCellsAttr, ArrayAttr::get(rewriter.getContext(), {}),
            ArrayAttr::get(rewriter.getContext(), {}));
      } else
        llvm_unreachable("Unknown scheduleable");
    }
    return success();
  }

  /// Schedules a block by inserting a branch argument assignment block (if any)
  /// before recursing into the scheduling of the block innards.
  /// Blocks 'from' and 'to' refer to blocks in the source program.
  /// parentCtrlBlock refers to the control block wherein control operations are
  /// to be inserted.
  LogicalResult schedulePath(PatternRewriter &rewriter,
                             const DenseSet<Block *> &path, Location loc,
                             Block *from, Block *to,
                             Block *parentCtrlBlock) const {
    /// Schedule any registered block arguments to be executed before the body
    /// of the branch.
    rewriter.setInsertionPointToEnd(parentCtrlBlock);
    auto preSeqOp = rewriter.create<calyx::SeqOp>(loc);
    rewriter.setInsertionPointToEnd(preSeqOp.getBodyBlock());
    for (auto barg :
         getState<ComponentLoweringState>().getBlockArgGroups(from, to))
      rewriter.create<calyx::EnableOp>(barg.getLoc(), barg.getSymName());

    return buildCFGControl(path, rewriter, parentCtrlBlock, from, to);
  }

  LogicalResult buildCFGControl(DenseSet<Block *> path,
                                PatternRewriter &rewriter,
                                mlir::Block *parentCtrlBlock,
                                mlir::Block *preBlock,
                                mlir::Block *block) const {
    if (path.count(block) != 0)
      return preBlock->getTerminator()->emitError()
             << "CFG backedge detected. Loops must be raised to 'scf.while' or "
                "'scf.for' operations.";

    rewriter.setInsertionPointToEnd(parentCtrlBlock);
    LogicalResult bbSchedResult =
        scheduleBasicBlock(rewriter, path, parentCtrlBlock, block);
    if (bbSchedResult.failed())
      return bbSchedResult;

    path.insert(block);
    auto successors = block->getSuccessors();
    auto nSuccessors = successors.size();
    if (nSuccessors > 0) {
      auto brOp = dyn_cast<BranchOpInterface>(block->getTerminator());
      assert(brOp);
      if (nSuccessors > 1) {
        /// TODO(mortbopet): we could choose to support ie. std.switch, but it
        /// would probably be easier to just require it to be lowered
        /// beforehand.
        assert(nSuccessors == 2 &&
               "only conditional branches supported for now...");
        /// Wrap each branch inside an if/else.
        auto cond = brOp->getOperand(0);
        auto condGroup = getState<ComponentLoweringState>()
                             .getEvaluatingGroup<calyx::CombGroupOp>(cond);
        auto symbolAttr = FlatSymbolRefAttr::get(
            StringAttr::get(getContext(), condGroup.getSymName()));

        auto ifOp = rewriter.create<calyx::IfOp>(
            brOp->getLoc(), cond, symbolAttr, /*initializeElseBody=*/true);
        rewriter.setInsertionPointToStart(ifOp.getThenBody());
        auto thenSeqOp = rewriter.create<calyx::SeqOp>(brOp.getLoc());
        rewriter.setInsertionPointToStart(ifOp.getElseBody());
        auto elseSeqOp = rewriter.create<calyx::SeqOp>(brOp.getLoc());

        bool trueBrSchedSuccess =
            schedulePath(rewriter, path, brOp.getLoc(), block, successors[0],
                         thenSeqOp.getBodyBlock())
                .succeeded();
        bool falseBrSchedSuccess = true;
        if (trueBrSchedSuccess) {
          falseBrSchedSuccess =
              schedulePath(rewriter, path, brOp.getLoc(), block, successors[1],
                           elseSeqOp.getBodyBlock())
                  .succeeded();
        }

        return success(trueBrSchedSuccess && falseBrSchedSuccess);
      } else {
        /// Schedule sequentially within the current parent control block.
        return schedulePath(rewriter, path, brOp.getLoc(), block,
                            successors.front(), parentCtrlBlock);
      }
    }
    return success();
  }

  // Insert a Par of initGroups at Location loc. Used as helper for
  // `buildWhileCtrlOp` and `buildForCtrlOp`.
  void
  insertParInitGroups(PatternRewriter &rewriter, Location loc,
                      const SmallVector<calyx::GroupOp> &initGroups) const {
    PatternRewriter::InsertionGuard g(rewriter);
    auto parOp = rewriter.create<calyx::ParOp>(loc);
    rewriter.setInsertionPointToStart(parOp.getBodyBlock());
    for (calyx::GroupOp group : initGroups)
      rewriter.create<calyx::EnableOp>(group.getLoc(), group.getName());
  }

  calyx::WhileOp buildWhileCtrlOp(ScfWhileOp whileOp,
                                  SmallVector<calyx::GroupOp> initGroups,
                                  PatternRewriter &rewriter) const {
    Location loc = whileOp.getLoc();
    /// Insert while iter arg initialization group(s). Emit a
    /// parallel group to assign one or more registers all at once.
    insertParInitGroups(rewriter, loc, initGroups);

    /// Insert the while op itself.
    auto cond = whileOp.getConditionValue();
    auto condGroup = getState<ComponentLoweringState>()
                         .getEvaluatingGroup<calyx::CombGroupOp>(cond);
    auto symbolAttr = FlatSymbolRefAttr::get(
        StringAttr::get(getContext(), condGroup.getSymName()));
    return rewriter.create<calyx::WhileOp>(loc, cond, symbolAttr);
  }

  calyx::RepeatOp buildForCtrlOp(ScfForOp forOp,
                                 SmallVector<calyx::GroupOp> const &initGroups,
                                 uint64_t bound,
                                 PatternRewriter &rewriter) const {
    Location loc = forOp.getLoc();
    // Insert for iter arg initialization group(s). Emit a
    // parallel group to assign one or more registers all at once.
    insertParInitGroups(rewriter, loc, initGroups);

    // Insert the repeatOp that corresponds to the For loop.
    return rewriter.create<calyx::RepeatOp>(loc, bound);
  }
};

/// LateSSAReplacement contains various functions for replacing SSA values that
/// were not replaced during op construction.
class LateSSAReplacement : public calyx::FuncOpPartialLoweringPattern {
  using FuncOpPartialLoweringPattern::FuncOpPartialLoweringPattern;

  LogicalResult partiallyLowerFuncToComp(FuncOp funcOp,
                                         PatternRewriter &) const override {
    funcOp.walk([&](scf::IfOp op) {
      for (auto res : getState<ComponentLoweringState>().getResultRegs(op))
        op.getOperation()->getResults()[res.first].replaceAllUsesWith(
            res.second.getOut());
    });

    funcOp.walk([&](scf::WhileOp op) {
      /// The yielded values returned from the while op will be present in the
      /// iterargs registers post execution of the loop.
      /// This is done now, as opposed to during BuildWhileGroups since if the
      /// results of the whileOp were replaced before
      /// BuildOpGroups/BuildControl, the whileOp would get dead-code
      /// eliminated.
      ScfWhileOp whileOp(op);
      for (auto res :
           getState<ComponentLoweringState>().getWhileLoopIterRegs(whileOp))
        whileOp.getOperation()->getResults()[res.first].replaceAllUsesWith(
            res.second.getOut());
    });

    funcOp.walk([&](memref::LoadOp loadOp) {
      if (calyx::singleLoadFromMemory(loadOp)) {
        /// In buildOpGroups we did not replace loadOp's results, to ensure a
        /// link between evaluating groups (which fix the input addresses of a
        /// memory op) and a readData result. Now, we may replace these SSA
        /// values with their memoryOp readData output.
        loadOp.getResult().replaceAllUsesWith(
            getState<ComponentLoweringState>()
                .getMemoryInterface(loadOp.getMemref())
                .readData());
      }
    });

    return success();
  }
};

/// Erases FuncOp operations.
class CleanupFuncOps : public calyx::FuncOpPartialLoweringPattern {
  using FuncOpPartialLoweringPattern::FuncOpPartialLoweringPattern;

  LogicalResult matchAndRewrite(FuncOp funcOp,
                                PatternRewriter &rewriter) const override {
    rewriter.eraseOp(funcOp);
    return success();
  }

  LogicalResult
  partiallyLowerFuncToComp(FuncOp funcOp,
                           PatternRewriter &rewriter) const override {
    return success();
  }
};

} // namespace scftocalyx

namespace {

using namespace circt::scftocalyx;

//===----------------------------------------------------------------------===//
// Pass driver
//===----------------------------------------------------------------------===//
class SCFToCalyxPass : public circt::impl::SCFToCalyxBase<SCFToCalyxPass> {
public:
  SCFToCalyxPass()
      : SCFToCalyxBase<SCFToCalyxPass>(), partialPatternRes(success()) {}
  void runOnOperation() override;

  LogicalResult setTopLevelFunction(mlir::ModuleOp moduleOp,
                                    std::string &topLevelFunction) {
    if (!topLevelFunctionOpt.empty()) {
      if (SymbolTable::lookupSymbolIn(moduleOp, topLevelFunctionOpt) ==
          nullptr) {
        moduleOp.emitError() << "Top level function '" << topLevelFunctionOpt
                             << "' not found in module.";
        return failure();
      }
      topLevelFunction = topLevelFunctionOpt;
    } else {
      /// No top level function set; infer top level if the module only contains
      /// a single function, else, throw error.
      auto funcOps = moduleOp.getOps<FuncOp>();
      if (std::distance(funcOps.begin(), funcOps.end()) == 1)
        topLevelFunction = (*funcOps.begin()).getSymName().str();
      else {
        moduleOp.emitError()
            << "Module contains multiple functions, but no top level "
               "function was set. Please see --top-level-function";
        return failure();
      }
    }

    return createOptNewTopLevelFn(moduleOp, topLevelFunction);
  }

  struct LoweringPattern {
    enum class Strategy { Once, Greedy };
    RewritePatternSet pattern;
    Strategy strategy;
  };

  //// Labels the entry point of a Calyx program.
  /// Furthermore, this function performs validation on the input function,
  /// to ensure that we've implemented the capabilities necessary to convert
  /// it.
  LogicalResult labelEntryPoint(StringRef topLevelFunction) {
    // Program legalization - the partial conversion driver will not run
    // unless some pattern is provided - provide a dummy pattern.
    struct DummyPattern : public OpRewritePattern<mlir::ModuleOp> {
      using OpRewritePattern::OpRewritePattern;
      LogicalResult matchAndRewrite(mlir::ModuleOp,
                                    PatternRewriter &) const override {
        return failure();
      }
    };

    ConversionTarget target(getContext());
    target.addLegalDialect<calyx::CalyxDialect>();
    target.addLegalDialect<scf::SCFDialect>();
    target.addIllegalDialect<hw::HWDialect>();
    target.addIllegalDialect<comb::CombDialect>();

    // Only accept std operations which we've added lowerings for
    target.addIllegalDialect<FuncDialect>();
    target.addIllegalDialect<ArithDialect>();
    target.addLegalOp<AddIOp, SelectOp, SubIOp, CmpIOp, ShLIOp, ShRUIOp,
                      ShRSIOp, AndIOp, XOrIOp, OrIOp, ExtUIOp, TruncIOp,
                      CondBranchOp, BranchOp, MulIOp, DivUIOp, DivSIOp, RemUIOp,
                      RemSIOp, ReturnOp, arith::ConstantOp, IndexCastOp, FuncOp,
                      ExtSIOp, CallOp, AddFOp, MulFOp, CmpFOp>();

    RewritePatternSet legalizePatterns(&getContext());
    legalizePatterns.add<DummyPattern>(&getContext());
    DenseSet<Operation *> legalizedOps;
    if (applyPartialConversion(getOperation(), target,
                               std::move(legalizePatterns))
            .failed())
      return failure();

    // Program conversion
    return calyx::applyModuleOpConversion(getOperation(), topLevelFunction);
  }

  /// 'Once' patterns are expected to take an additional LogicalResult&
  /// argument, to forward their result state (greedyPatternRewriteDriver
  /// results are skipped for Once patterns).
  template <typename TPattern, typename... PatternArgs>
  void addOncePattern(SmallVectorImpl<LoweringPattern> &patterns,
                      PatternArgs &&...args) {
    RewritePatternSet ps(&getContext());
    ps.add<TPattern>(&getContext(), partialPatternRes, args...);
    patterns.push_back(
        LoweringPattern{std::move(ps), LoweringPattern::Strategy::Once});
  }

  template <typename TPattern, typename... PatternArgs>
  void addGreedyPattern(SmallVectorImpl<LoweringPattern> &patterns,
                        PatternArgs &&...args) {
    RewritePatternSet ps(&getContext());
    ps.add<TPattern>(&getContext(), args...);
    patterns.push_back(
        LoweringPattern{std::move(ps), LoweringPattern::Strategy::Greedy});
  }

  LogicalResult runPartialPattern(RewritePatternSet &pattern, bool runOnce) {
    assert(pattern.getNativePatterns().size() == 1 &&
           "Should only apply 1 partial lowering pattern at once");

    // During component creation, the function body is inlined into the
    // component body for further processing. However, proper control flow
    // will only be established later in the conversion process, so ensure
    // that rewriter optimizations (especially DCE) are disabled.
    GreedyRewriteConfig config;
    config.enableRegionSimplification =
        mlir::GreedySimplifyRegionLevel::Disabled;
    if (runOnce)
      config.maxIterations = 1;

    /// Can't return applyPatternsAndFoldGreedily. Root isn't
    /// necessarily erased so it will always return failed(). Instead,
    /// forward the 'succeeded' value from PartialLoweringPatternBase.
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(pattern),
                                       config);
    return partialPatternRes;
  }

private:
  LogicalResult partialPatternRes;
  std::shared_ptr<calyx::CalyxLoweringState> loweringState = nullptr;

  /// Creates a new new top-level function based on `baseName`.
  FuncOp createNewTopLevelFn(ModuleOp moduleOp, std::string &baseName) {
    std::string newName = baseName;
    unsigned counter = 0;
    while (SymbolTable::lookupSymbolIn(moduleOp, newName)) {
      newName = llvm::join_items("_", baseName, std::to_string(++counter));
    }

    OpBuilder builder(moduleOp.getContext());
    builder.setInsertionPointToStart(moduleOp.getBody());

    FunctionType funcType = builder.getFunctionType({}, {});

    if (auto newFunc =
            builder.create<FuncOp>(moduleOp.getLoc(), newName, funcType)) {
      baseName = newName;
      return newFunc;
    }

    moduleOp.emitError("Cannot create new top-level function.");

    return nullptr;
  }

  /// Insert a call from the newly created top-level function/`caller` to the
  /// old top-level function/`callee`; and create `memref.alloc`s inside the new
  /// top-level function for arguments with `memref` types and for the
  /// `memref.alloc`s inside `callee`.
  void insertCallFromNewTopLevel(OpBuilder &builder, FuncOp caller, FuncOp callee) {
    if (caller.getBody().empty()) {
      caller.addEntryBlock();
    }

    Block *callerEntryBlock = &caller.getBody().front();
    builder.setInsertionPointToStart(callerEntryBlock);
    
    // First, add arguments to the new top-level function.
    // For those memref arguments passing to the old top-level function, 
    // we simply create `memref.allocOp`s, which will be lowered to `@external` memory cells.
    // On the contrary, the new top-level function should take those non-memref type arguments
    // since we still need to pass them.
    SmallVector<Type, 4> nonMemRefCalleeArgTypes;
    for (auto arg : callee.getArguments()) {
      if (!isa<MemRefType>(arg.getType())) {
        nonMemRefCalleeArgTypes.push_back(arg.getType());
      }
    }
    for (Type type : nonMemRefCalleeArgTypes)
      callerEntryBlock->addArgument(type, caller.getLoc());
    
    // After inserting additional arguments, we update its function type to reflect this.
    FunctionType callerFnType = caller.getFunctionType();
    SmallVector<Type, 4> updatedCallerArgTypes(caller.getFunctionType().getInputs());
    updatedCallerArgTypes.append(nonMemRefCalleeArgTypes.begin(),
                                 nonMemRefCalleeArgTypes.end());
    caller.setType(FunctionType::get(caller.getContext(), updatedCallerArgTypes,
                                     callerFnType.getResults()));
    
    // Extra memref arguments to pass from the new top-level to the old one are created
    // whenener there is an external memory declaration operation in the old top-level.
    // Those would have been `@external` memory cells will now become `ref` cells passed
    // as arguments from the new top-level to the old one.
    Block *calleeFnBody = &callee.getBody().front();
    unsigned originalCalleeArgNum = callee.getArguments().size();
    builder.setInsertionPointToStart(calleeFnBody);
    SmallVector<Value, 4> extraMemRefArgs;
    SmallVector<Type, 4> extraMemRefArgTypes;
    SmallVector<Value, 4> extraMemRefOperands;
    SmallVector<Operation *, 4> opsToModify;
    for (auto &op : callee.getBody().getOps()) {
      if (isa<memref::AllocaOp>(op) || isa<memref::AllocOp>(op) || isa<memref::GetGlobalOp>(op))
        opsToModify.push_back(&op);
    }

    builder.setInsertionPointToEnd(callerEntryBlock);
    for (auto *op : opsToModify) {
        Value newOpRes;
        if (auto allocaOp = dyn_cast<memref::AllocaOp>(op)) {
          newOpRes = builder.create<memref::AllocaOp>(callee.getLoc(), allocaOp.getType());
        }
        else if (auto allocOp = dyn_cast<memref::AllocOp>(op)) {
          newOpRes = builder.create<memref::AllocOp>(callee.getLoc(), allocOp.getType());
        } else if (auto getGlobalOp = dyn_cast<memref::GetGlobalOp>(op)) {
          newOpRes = builder.create<memref::GetGlobalOp>(caller.getLoc(), getGlobalOp.getType(), getGlobalOp.getName());
        }
        extraMemRefOperands.push_back(newOpRes); 

        calleeFnBody->addArgument(newOpRes.getType(), callee.getLoc());
        BlockArgument newBodyArg = calleeFnBody->getArguments().back();
        op->getResult(0).replaceAllUsesWith(newBodyArg);
        op->erase();
        extraMemRefArgs.push_back(newBodyArg);
        extraMemRefArgTypes.push_back(newBodyArg.getType());
    }
    
    SmallVector<Type, 4> updatedCalleeArgTypes(callee.getFunctionType().getInputs());
    updatedCalleeArgTypes.append(extraMemRefArgTypes.begin(), extraMemRefArgTypes.end());
    callee.setType(FunctionType::get(callee.getContext(), updatedCalleeArgTypes, callee.getFunctionType().getResults()));

    // After we have updated old top-level function's type, we can create `memref.allocOp` in the
    // body of the new to-level; and identify the type of the function signature of the `callOp`.
    unsigned otherArgsCount = 0;
    SmallVector<Value, 4> calleeArgFnOperands;
    builder.setInsertionPointToStart(callerEntryBlock);
    for (auto arg : callee.getArguments().take_front(originalCalleeArgNum)) {
      if (isa<MemRefType>(arg.getType())) {
        auto memrefType = cast<MemRefType>(arg.getType());
        auto allocOp = builder.create<memref::AllocOp>(callee.getLoc(), memrefType);
        calleeArgFnOperands.push_back(allocOp);
      } else {
        auto callerArg = callerEntryBlock->getArgument(otherArgsCount++);
        calleeArgFnOperands.push_back(callerArg);
      }
    }
    
    SmallVector<Value, 4> fnOperands;
    fnOperands.append(calleeArgFnOperands.begin(), calleeArgFnOperands.end());
    fnOperands.append(extraMemRefOperands.begin(), extraMemRefOperands.end());
    auto calleeName =
        SymbolRefAttr::get(builder.getContext(), callee.getSymName());
    auto resultTypes = callee.getResultTypes();
    
    builder.setInsertionPointToEnd(callerEntryBlock);
    builder.create<CallOp>(caller.getLoc(), calleeName, resultTypes, fnOperands);
  }

  /// Conditionally creates an optional new top-level function; and inserts a
  /// call from the new top-level function to the old top-level function if we
  /// did create one
  LogicalResult createOptNewTopLevelFn(ModuleOp moduleOp,
                                       std::string &topLevelFunction) {
    auto hasMemrefArguments = [](FuncOp func) {
      return std::any_of(
          func.getArguments().begin(), func.getArguments().end(),
          [](BlockArgument arg) { return isa<MemRefType>(arg.getType()); });
    };

    /// We only create a new top-level function and call the original top-level
    /// function from the new one if the original top-level has `memref` in its
    /// argument
    bool hasMemrefArgsInTopLevel = false;
    for (auto funcOp : moduleOp.getOps<FuncOp>()) {
      if (funcOp.getName() == topLevelFunction) {
        if (hasMemrefArguments(funcOp)) {
          hasMemrefArgsInTopLevel = true;
        }
      }
    }

    std::string oldName = topLevelFunction;
    if (hasMemrefArgsInTopLevel) {
      auto newTopLevelFunc = createNewTopLevelFn(moduleOp, topLevelFunction);

      OpBuilder builder(moduleOp.getContext());
      Operation *oldTopLevelFuncOp =
          SymbolTable::lookupSymbolIn(moduleOp, oldName);
      auto oldTopLevelFunc = dyn_cast_or_null<FuncOp>(oldTopLevelFuncOp);

      if (!oldTopLevelFunc)
        oldTopLevelFunc.emitOpError("Original top-level function not found!");

      insertCallFromNewTopLevel(builder, newTopLevelFunc, oldTopLevelFunc);
    }

    return success();
  }
};

void SCFToCalyxPass::runOnOperation() {
  // Clear internal state. See https://github.com/llvm/circt/issues/3235
  loweringState.reset();
  partialPatternRes = LogicalResult::failure();

  std::string topLevelFunction;
  if (failed(setTopLevelFunction(getOperation(), topLevelFunction))) {
    signalPassFailure();
    return;
  }

  /// Start conversion
  if (failed(labelEntryPoint(topLevelFunction))) {
    signalPassFailure();
    return;
  }
  loweringState = std::make_shared<calyx::CalyxLoweringState>(getOperation(),
                                                              topLevelFunction);

  /// --------------------------------------------------------------------------
  /// If you are a developer, it may be helpful to add a
  /// 'getOperation()->dump()' call after the execution of each stage to
  /// view the transformations that's going on.
  /// --------------------------------------------------------------------------

  /// A mapping is maintained between a function operation and its corresponding
  /// Calyx component.
  DenseMap<FuncOp, calyx::ComponentOp> funcMap;
  SmallVector<LoweringPattern, 8> loweringPatterns;
  calyx::PatternApplicationState patternState;

  /// Replace memory accesses with their corresponding banks if the user has
  /// specified the number of available banks.
  addOncePattern<MemoryBanking>(loweringPatterns, patternState, funcMap,
                                *loweringState);

  /// Creates a new Calyx component for each FuncOp in the inpurt module.
  addOncePattern<FuncOpConversion>(loweringPatterns, patternState, funcMap,
                                   *loweringState);

  /// This pass inlines scf.ExecuteRegionOp's by adding control-flow.
  addGreedyPattern<InlineExecuteRegionOpPattern>(loweringPatterns);

  addOncePattern<BuildSwitchGroups>(loweringPatterns, patternState, funcMap,
                                    *loweringState);

  /// This pattern converts all index typed values to an i32 integer.
  addOncePattern<calyx::ConvertIndexTypes>(loweringPatterns, patternState,
                                           funcMap, *loweringState);

  /// This pattern creates registers for all basic-block arguments.
  addOncePattern<calyx::BuildBasicBlockRegs>(loweringPatterns, patternState,
                                             funcMap, *loweringState);

  addOncePattern<calyx::BuildCallInstance>(loweringPatterns, patternState,
                                           funcMap, *loweringState);

  /// This pattern creates registers for the function return values.
  addOncePattern<calyx::BuildReturnRegs>(loweringPatterns, patternState,
                                         funcMap, *loweringState);

  /// This pattern creates registers for iteration arguments of scf.while
  /// operations. Additionally, creates a group for assigning the initial
  /// value of the iteration argument registers.
  addOncePattern<BuildWhileGroups>(loweringPatterns, patternState, funcMap,
                                   *loweringState);

  /// This pattern creates registers for iteration arguments of scf.for
  /// operations. Additionally, creates a group for assigning the initial
  /// value of the iteration argument registers.
  addOncePattern<BuildForGroups>(loweringPatterns, patternState, funcMap,
                                 *loweringState);

  addOncePattern<BuildIfGroups>(loweringPatterns, patternState, funcMap,
                                *loweringState);
  /// This pattern converts operations within basic blocks to Calyx library
  /// operators. Combinational operations are assigned inside a
  /// calyx::CombGroupOp, and sequential inside calyx::GroupOps.
  /// Sequential groups are registered with the Block* of which the operation
  /// originated from. This is used during control schedule generation. By
  /// having a distinct group for each operation, groups are analogous to SSA
  /// values in the source program.
  addOncePattern<BuildOpGroups>(loweringPatterns, patternState, funcMap,
                                *loweringState);

  /// This pattern traverses the CFG of the program and generates a control
  /// schedule based on the calyx::GroupOp's which were registered for each
  /// basic block in the source function.
  addOncePattern<BuildControl>(loweringPatterns, patternState, funcMap,
                               *loweringState);

  /// This pass recursively inlines use-def chains of combinational logic (from
  /// non-stateful groups) into groups referenced in the control schedule.
  addOncePattern<calyx::InlineCombGroups>(loweringPatterns, patternState,
                                          *loweringState);

  /// This pattern performs various SSA replacements that must be done
  /// after control generation.
  addOncePattern<LateSSAReplacement>(loweringPatterns, patternState, funcMap,
                                     *loweringState);

  /// Eliminate any unused combinational groups. This is done before
  /// calyx::RewriteMemoryAccesses to avoid inferring slice components for
  /// groups that will be removed.
  addGreedyPattern<calyx::EliminateUnusedCombGroups>(loweringPatterns);

  /// This pattern rewrites accesses to memories which are too wide due to
  /// index types being converted to a fixed-width integer type.
  addOncePattern<calyx::RewriteMemoryAccesses>(loweringPatterns, patternState,
                                               *loweringState);

  /// This pattern removes the source FuncOp which has now been converted into
  /// a Calyx component.
  addOncePattern<CleanupFuncOps>(loweringPatterns, patternState, funcMap,
                                 *loweringState);

  /// Sequentially apply each lowering pattern.
  for (auto &pat : loweringPatterns) {
    LogicalResult partialPatternRes = runPartialPattern(
        pat.pattern,
        /*runOnce=*/pat.strategy == LoweringPattern::Strategy::Once);
    if (succeeded(partialPatternRes))
      continue;
    signalPassFailure();
    return;
  }

  //===--------------------------------------------------------------------===//
  // Cleanup patterns
  //===--------------------------------------------------------------------===//
  RewritePatternSet cleanupPatterns(&getContext());
  cleanupPatterns.add<calyx::MultipleGroupDonePattern,
                      calyx::NonTerminatingGroupDonePattern>(&getContext());
  if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                          std::move(cleanupPatterns)))) {
    signalPassFailure();
    return;
  }

  if (ciderSourceLocationMetadata) {
    // Debugging information for the Cider debugger.
    // Reference: https://docs.calyxir.org/debug/cider.html
    SmallVector<Attribute, 16> sourceLocations;
    getOperation()->walk([&](calyx::ComponentOp component) {
      return getCiderSourceLocationMetadata(component, sourceLocations);
    });

    MLIRContext *context = getOperation()->getContext();
    getOperation()->setAttr("calyx.metadata",
                            ArrayAttr::get(context, sourceLocations));
  }
}
} // namespace

//===----------------------------------------------------------------------===//
// Pass initialization
//===----------------------------------------------------------------------===//

std::unique_ptr<OperationPass<ModuleOp>> createSCFToCalyxPass() {
  return std::make_unique<SCFToCalyxPass>();
}

} // namespace circt
