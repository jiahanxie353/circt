//===- DebugTypes.td - Debug dialect types -----------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_DEBUG_DEBUGTYPES_TD
#define CIRCT_DIALECT_DEBUG_DEBUGTYPES_TD

include "circt/Dialect/Debug/DebugDialect.td"
include "mlir/IR/AttrTypeBase.td"

class DebugTypeDef<string name> : TypeDef<DebugDialect, name> { }

def StructType : DebugTypeDef<"Struct"> {
  let mnemonic = "struct";
}

def ArrayType : DebugTypeDef<"Array"> {
  let mnemonic = "array";
}

#endif // CIRCT_DIALECT_DEBUG_DEBUGTYPES_TD
