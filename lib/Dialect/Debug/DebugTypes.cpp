//===- DebugTypes.cpp - Debug dialect types -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Debug/DebugTypes.h"
#include "circt/Dialect/Debug/DebugDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace debug;
using namespace mlir;

//===----------------------------------------------------------------------===//
// StructType
//===----------------------------------------------------------------------===//

// Type StructType::parse(AsmParser &parser) {
//   SmallVector<Type> fields;
//   if (failed(parser.parseCommaSeparatedList(
//           AsmParser::Delimiter::LessGreater,
//           [&] { return parser.parseType(fields.emplace_back()); })))
//     return {};
//   return StructType::get(parser.getContext(), fields);
// }

// void StructType::print(AsmPrinter &printer) const {
//   printer << '<' << getFields() << '>';
// }

//===----------------------------------------------------------------------===//
// Table Gen
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "circt/Dialect/Debug/DebugTypes.cpp.inc"

void DebugDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "circt/Dialect/Debug/DebugTypes.cpp.inc"
      >();
}
