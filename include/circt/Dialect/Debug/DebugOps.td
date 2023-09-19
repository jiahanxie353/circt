//===- DebugOps.td - Debug dialect operations ====----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_DEBUG_DEBUGOPS_TD
#define CIRCT_DIALECT_DEBUG_DEBUGOPS_TD

include "circt/Dialect/Debug/DebugDialect.td"
include "circt/Dialect/Debug/DebugTypes.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

class DebugOp<string mnemonic, list<Trait> traits = []> :
  Op<DebugDialect, mnemonic, traits>;

def VariableOp : DebugOp<"variable"> {
  let summary = "A port, constant, or local variable";
  let arguments = (ins StrAttr:$name, AnyType:$value);
  let assemblyFormat = [{ $name `,` $value attr-dict `:` type($value) }];
}

def StructOp : DebugOp<"struct", [
  Pure,
  PredOpTrait<"number of fields and names match",
    CPred<"$fields.size() == $names.size()">>
]> {
  let summary = "Aggregate values into a struct";
  let description = [{
    Creates a struct aggregate from a list of names and values, to be used in
    debug information.
  }];
  let arguments = (ins Variadic<AnyType>:$fields, StrArrayAttr:$names);
  let results = (outs StructType:$result);
  let hasCustomAssemblyFormat = 1;
}

def ArrayOp : DebugOp<"array", [Pure]> {
  let summary = "Aggregate values into an array";
  let description = [{
    Creates an array aggregate from a list of values, to be used in debug
    information. The first operand is placed at array index 0. The last operand
    is placed at the highest array index.
  }];
  let arguments = (ins Variadic<AnyType>:$elements);
  let results = (outs ArrayType:$result);
  let hasCustomAssemblyFormat = 1;
}

#endif // CIRCT_DIALECT_DEBUG_DEBUGOPS_TD
