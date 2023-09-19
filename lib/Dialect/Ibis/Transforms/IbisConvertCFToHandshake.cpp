//===- IbisConvertCFToHandshakePass.cpp -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"

#include "circt/Dialect/Ibis/IbisDialect.h"
#include "circt/Dialect/Ibis/IbisOps.h"
#include "circt/Dialect/Ibis/IbisPasses.h"
#include "circt/Dialect/Ibis/IbisTypes.h"

#include "circt/Transforms/Passes.h"
#include "mlir/Transforms/DialectConversion.h"

#include "circt/Conversion/CFToHandshake.h"

using namespace mlir;
using namespace circt;
using namespace ibis;

namespace {

struct ConvertCFToHandshakePass
    : public IbisConvertCFToHandshakeBase<ConvertCFToHandshakePass> {
  void runOnOperation() override;
};
} // anonymous namespace

void ConvertCFToHandshakePass::runOnOperation() {
  MethodOp method = getOperation();

  // Add a control input/output to the method.
  handshake::HandshakeLowering fol(method.getBody());
  if (failed(handshake::lowerRegion<ibis::ReturnOp, ibis::ReturnOp>(
          fol,
          /*sourceConstants*/ false, /*disableTaskPipelining*/ false)))
    return signalPassFailure();
}

std::unique_ptr<Pass> circt::ibis::createConvertCFToHandshakePass() {
  return std::make_unique<ConvertCFToHandshakePass>();
}
