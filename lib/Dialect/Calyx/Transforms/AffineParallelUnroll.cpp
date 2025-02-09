//===- AffineParallelUnroll.cpp - Unroll AffineParallelOp --------------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Unroll AffineParallelOp to facilitate lowering to Calyx ParOp.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Calyx/CalyxPasses.h"
#include "circt/Support/LLVM.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"

namespace circt {
namespace calyx {
#define GEN_PASS_DEF_AFFINEPARALLELUNROLL
#include "circt/Dialect/Calyx/CalyxPasses.h.inc"
} // namespace calyx
} // namespace circt

using namespace circt;
using namespace mlir;
using namespace mlir::affine;
using namespace mlir::arith;

namespace {

struct AffineParallelUnroll : public OpConversionPattern<AffineParallelOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AffineParallelOp affineParallelOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!affineParallelOp.getResults().empty()) {
      affineParallelOp.emitError(
          "affine.parallel with reductions is not supported yet");
      return failure();
    }

    Location loc = affineParallelOp.getLoc();

    rewriter.setInsertionPointAfter(affineParallelOp);
    // Create a single-iteration for loop op and mark its specialty by setting
    // the attribute.
    AffineForOp affineForOp = rewriter.create<AffineForOp>(loc, 0, 1, 1);
    affineForOp->setAttr("calyx.parallel", rewriter.getBoolAttr(true));

    SmallVector<int64_t> pLoopLowerBounds =
        affineParallelOp.getLowerBoundsMap().getConstantResults();
    if (pLoopLowerBounds.empty()) {
      affineParallelOp.emitError(
          "affine.parallel must have constant lower bounds");
      return failure();
    }
    SmallVector<int64_t> pLoopUpperBounds =
        affineParallelOp.getUpperBoundsMap().getConstantResults();
    if (pLoopUpperBounds.empty()) {
      affineParallelOp.emitError(
          "affine.parallel must have constant upper bounds");
      return failure();
    }
    SmallVector<int64_t, 8> pLoopSteps = affineParallelOp.getSteps();

    Block *pLoopBody = affineParallelOp.getBody();
    MutableArrayRef<BlockArgument> pLoopIVs = affineParallelOp.getIVs();

    OpBuilder insideBuilder(affineForOp);
    SmallVector<int64_t> indices = pLoopLowerBounds;
    while (true) {
      insideBuilder.setInsertionPointToStart(affineForOp.getBody());
      // Create an `scf.execute_region` to wrap each unrolled block since
      // `affine.parallel` requires only one block in the body region.
      auto executeRegionOp =
          insideBuilder.create<scf::ExecuteRegionOp>(loc, TypeRange{});
      Region &executeRegionRegion = executeRegionOp.getRegion();
      Block *executeRegionBlock = &executeRegionRegion.emplaceBlock();

      OpBuilder regionBuilder(executeRegionOp);
      // Each iteration starts with a fresh mapping, so each new block’s
      // argument of a region-based operation (such as `affine.for`) get
      // re-mapped independently.
      IRMapping operandMap;
      regionBuilder.setInsertionPointToEnd(executeRegionBlock);
      // Map induction variables to constant indices
      for (unsigned i = 0; i < indices.size(); ++i) {
        Value ivConstant =
            regionBuilder.create<arith::ConstantIndexOp>(loc, indices[i]);
        operandMap.map(pLoopIVs[i], ivConstant);
      }

      for (auto it = pLoopBody->begin(); it != std::prev(pLoopBody->end());
           ++it)
        regionBuilder.clone(*it, operandMap);

      // A terminator should always be inserted in `scf.execute_region`'s block.
      regionBuilder.create<scf::YieldOp>(loc);

      // Increment indices using `step`.
      bool done = false;
      for (int dim = indices.size() - 1; dim >= 0; --dim) {
        indices[dim] += pLoopSteps[dim];
        if (indices[dim] < pLoopUpperBounds[dim])
          break;
        indices[dim] = pLoopLowerBounds[dim];
        if (dim == 0)
          // All combinations have been generated
          done = true;
      }
      if (done)
        break;
    }

    rewriter.replaceOp(affineParallelOp, affineForOp);

    return success();
  }
};

struct AffineParallelUnrollPass
    : public circt::calyx::impl::AffineParallelUnrollBase<
          AffineParallelUnrollPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::scf::SCFDialect>();
  }
  void runOnOperation() override;
};

} // end anonymous namespace

void AffineParallelUnrollPass::runOnOperation() {
  auto *ctx = &getContext();
  ConversionTarget target(*ctx);
  target.addLegalDialect<affine::AffineDialect>();
  target.addLegalDialect<scf::SCFDialect>();
  target.addLegalDialect<arith::ArithDialect>();
  target.addLegalOp<ModuleOp, func::FuncOp, func::ReturnOp>();
  target.addIllegalOp<AffineParallelOp>();

  RewritePatternSet patterns(ctx);
  patterns.add<AffineParallelUnroll>(ctx);

  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<mlir::Pass> circt::calyx::createAffineParallelUnrollPass() {
  return std::make_unique<AffineParallelUnrollPass>();
}
