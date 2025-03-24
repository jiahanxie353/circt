//===- PreSCFToCalyxCanonicalize.cpp
//----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Calyx/CalyxPasses.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace circt {
namespace calyx {
#define GEN_PASS_DEF_PRESCFTOCALYXCANONICALIZE
#include "circt/Dialect/Calyx/CalyxPasses.h.inc"
} // namespace calyx
} // namespace circt

using namespace mlir;
using namespace mlir::arith;
using namespace mlir::memref;
using namespace mlir::scf;
using namespace mlir::func;
using namespace circt;

namespace {
class PreSCFToCalyxCanonicalizePass
    : public circt::calyx::impl::PreSCFToCalyxCanonicalizeBase<
          PreSCFToCalyxCanonicalizePass> {
  void runOnOperation() override;
};
} // namespace

void PreSCFToCalyxCanonicalizePass::runOnOperation() {
  MLIRContext *ctx = &getContext();

  RewritePatternSet patterns(ctx);
  scf::IndexSwitchOp::getCanonicalizationPatterns(patterns, ctx);
  scf::IfOp::getCanonicalizationPatterns(patterns, ctx);

  arith::AddIOp::getCanonicalizationPatterns(patterns, ctx);
  arith::SubIOp::getCanonicalizationPatterns(patterns, ctx);
  arith::MulIOp::getCanonicalizationPatterns(patterns, ctx);
  arith::MulFOp::getCanonicalizationPatterns(patterns, ctx);
  arith::DivFOp::getCanonicalizationPatterns(patterns, ctx);
  arith::SelectOp::getCanonicalizationPatterns(patterns, ctx);
  arith::CmpIOp::getCanonicalizationPatterns(patterns, ctx);
  arith::CmpFOp::getCanonicalizationPatterns(patterns, ctx);
  arith::TruncIOp::getCanonicalizationPatterns(patterns, ctx);
  arith::XOrIOp::getCanonicalizationPatterns(patterns, ctx);
  arith::OrIOp::getCanonicalizationPatterns(patterns, ctx);
  arith::AndIOp::getCanonicalizationPatterns(patterns, ctx);
  arith::SelectOp::getCanonicalizationPatterns(patterns, ctx);
  arith::ConstantIndexOp::getCanonicalizationPatterns(patterns, ctx);
  arith::ConstantIntOp::getCanonicalizationPatterns(patterns, ctx);
  arith::ConstantFloatOp::getCanonicalizationPatterns(patterns, ctx);
  arith::ExtSIOp::getCanonicalizationPatterns(patterns, ctx);

  affine::AffineIfOp::getCanonicalizationPatterns(patterns, ctx);
  affine::AffineStoreOp::getCanonicalizationPatterns(patterns, ctx);
  affine::AffineLoadOp::getCanonicalizationPatterns(patterns, ctx);
  affine::AffineForOp::getCanonicalizationPatterns(patterns, ctx);
  affine::AffineApplyOp::getCanonicalizationPatterns(patterns, ctx);

  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
    getOperation()->emitError("Failed to apply canonicalization.");
    signalPassFailure();
  }

  ConversionTarget target(*ctx);
  target.addLegalDialect<arith::ArithDialect, memref::MemRefDialect,
                         scf::SCFDialect, affine::AffineDialect>();
}

std::unique_ptr<mlir::Pass>
circt::calyx::createPreSCFToCalyxCanonicalizePass() {
  return std::make_unique<PreSCFToCalyxCanonicalizePass>();
}
