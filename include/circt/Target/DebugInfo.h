//===- DebugInfo.h - Debug info emission ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares entry points to emit debug information.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_TARGET_DEBUGINFO_H
#define CIRCT_TARGET_DEBUGINFO_H

#include "circt/Support/LLVM.h"
#include "llvm/Support/raw_ostream.h"

namespace circt {
namespace debug {

/// Register all debug information emission flavors as from-MLIR translations.
void registerTranslations();

/// Dump the debug information in the given `module` in a human-readable format.
LogicalResult dumpDebugInfo(Operation *module, llvm::raw_ostream &os);

/// Serialize the debug information in the given `module` into the HGLDD format
/// and writes it to `output`.
LogicalResult emitHGLDD(Operation *module, StringRef directory,
                        llvm::raw_ostream &os);

/// Serialize the debug information in the given `module` into the HGLDD format
/// and emit one companion HGLDD file per emitted HDL file. This requires that
/// a prior emission pass such as `ExportVerilog` has annotated emission
/// locations on the operations in `module`.
LogicalResult emitSplitHGLDD(Operation *module, StringRef directory);

} // namespace debug
} // namespace circt

#endif // CIRCT_TARGET_DEBUGINFO_H
