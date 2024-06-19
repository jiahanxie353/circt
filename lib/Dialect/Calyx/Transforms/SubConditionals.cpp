//===- SubConditionals.cpp - Substitute Conditional Arithmetic Operations Pass
//---*- C++ -*-===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains the definitions of the conditional arithmetic substitution pass.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Calyx/CalyxHelpers.h"
#include "circt/Dialect/Calyx/CalyxOps.h"
#include "circt/Dialect/Calyx/CalyxPasses.h"
#include "circt/Support/LLVM.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"

namespace circt {
namespace calyx {
#define GEN_PASS_DEF_SUBCONDITIONALS
#include "circt/Dialect/Calyx/CalyxPasses.h.inc"
} // namespace calyx
} // namespace circt

using namespace circt;
using namespace calyx;
using namespace mlir;

namespace {

struct SubConditionalsPass : public circt::calyx::impl::SubConditionalsBase<SubConditionalsPass> {
  LogicalResult transformOp(OpBuilder &builder, arith::MaximumFOp maxFOp) {
    auto loc = maxFOp.getLoc();
    builder.setInsertionPointAfter(maxFOp);
    auto cmpFOp = builder.create<arith::CmpFOp>(
        loc, arith::CmpFPredicate::OGT, maxFOp.getLhs(), maxFOp.getRhs());
    auto ifOp = builder.create<scf::IfOp>(loc, maxFOp.getResult().getType(),
                                          cmpFOp.getResult(), true);

    auto &thenBlock = ifOp.getThenRegion().front();
    auto &elseBlock = ifOp.getElseRegion().front();

    builder.setInsertionPointToStart(&thenBlock);
    builder.create<scf::YieldOp>(loc, maxFOp.getLhs());

    builder.setInsertionPointToStart(&elseBlock);
    builder.create<scf::YieldOp>(loc, maxFOp.getRhs());

    maxFOp.replaceAllUsesWith(ifOp.getResults()[0]);
    maxFOp.erase();
    return success();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    auto builder = OpBuilder(funcOp);
    funcOp.walk([&](arith::MaximumFOp op) {
      if (failed(transformOp(builder, op))) {
        signalPassFailure();
      }
    });
  };
};

} // namespace

std::unique_ptr<mlir::Pass> circt::calyx::createSubConditionalsPass() {
  return std::make_unique<SubConditionalsPass>();
}
