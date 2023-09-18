//===- EmitHGLDD.cpp - HGLDD debug info emission --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Analysis/DebugInfo.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Target/DebugInfo.h"
#include "mlir/IR/Threading.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/IndentedOstream.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/ToolOutputFile.h"

#define DEBUG_TYPE "di"

using namespace mlir;
using namespace circt;
using llvm::MapVector;
using llvm::SmallMapVector;

static void findLocations(Location loc, unsigned level,
                          SmallVectorImpl<FileLineColLoc> &locs) {
  if (auto nameLoc = dyn_cast<NameLoc>(loc)) {
    if (nameLoc.getName() == "emitted")
      if (level-- == 0)
        return;
    findLocations(nameLoc.getChildLoc(), level, locs);
  } else if (auto fusedLoc = dyn_cast<FusedLoc>(loc)) {
    auto strAttr = dyn_cast_or_null<StringAttr>(fusedLoc.getMetadata());
    if (strAttr && strAttr.getValue() == "verilogLocations")
      if (level-- == 0)
        return;
    for (auto innerLoc : fusedLoc.getLocations())
      findLocations(innerLoc, level, locs);
  } else if (auto fileLoc = dyn_cast<FileLineColLoc>(loc)) {
    if (level == 0)
      locs.push_back(fileLoc);
  }
}

static FileLineColLoc findBestLocation(Location loc, bool emitted) {
  SmallVector<FileLineColLoc> locs;
  findLocations(loc, emitted ? 1 : 0, locs);
  for (auto loc : locs)
    if (!loc.getFilename().getValue().endswith(".fir"))
      return loc;
  for (auto loc : locs)
    if (loc.getFilename().getValue().endswith(".fir"))
      return loc;
  return {};
}

//===----------------------------------------------------------------------===//
// HGLDD File Emission
//===----------------------------------------------------------------------===//

namespace {

/// Contextual information for a single HGLDD file to be emitted.
struct FileEmitter {
  SmallVector<DIModule *> modules;
  SmallString<64> outputFileName;
  StringAttr hdlFile;
  SmallMapVector<StringAttr, unsigned, 8> sourceFiles;

  void emit(llvm::raw_ostream &os);
  void emit(llvm::json::OStream &json);
  void emitLoc(llvm::json::OStream &json, FileLineColLoc loc);
  void emitModule(llvm::json::OStream &json, DIModule *module);
  void emitInstance(llvm::json::OStream &json, DIInstance *instance);
  void emitVariable(llvm::json::OStream &json, DIVariable *variable);

  /// Get a numeric index for the given `sourceFile`. Populates `sourceFiles`
  /// with a unique ID assignment for each source file.
  unsigned getSourceFile(StringAttr sourceFile) {
    auto &slot = sourceFiles[sourceFile];
    if (slot == 0)
      slot = sourceFiles.size();
    return slot;
  }

  /// Find the best location and, if one is found, emit it under the given
  /// `fieldName`.
  void findAndEmitLoc(llvm::json::OStream &json, StringRef fieldName,
                      Location loc, bool emitted) {
    if (auto fileLoc = findBestLocation(loc, emitted))
      json.attributeObject(fieldName, [&] { emitLoc(json, fileLoc); });
  }
};

} // namespace

void FileEmitter::emit(llvm::raw_ostream &os) {
  llvm::json::OStream json(os, 2);
  emit(json);
  os << "\n";
}

void FileEmitter::emit(llvm::json::OStream &json) {
  // The "HGLDD" header field needs to be the first in the JSON file (which
  // violates the JSON spec, but what can you do). But we only know after module
  // emission what the contents of the header will be.
  std::string rawObjects;
  {
    llvm::raw_string_ostream objectsOS(rawObjects);
    llvm::json::OStream objectsJson(objectsOS, 2);
    objectsJson.arrayBegin(); // dummy for indentation
    objectsJson.arrayBegin();
    for (auto *module : modules)
      emitModule(objectsJson, module);
    objectsJson.arrayEnd();
    objectsJson.arrayEnd(); // dummy for indentation
  }

  std::optional<unsigned> hdlFileIndex;
  if (hdlFile)
    hdlFileIndex = getSourceFile(hdlFile);

  json.objectBegin();

  json.attributeObject("HGLDD", [&] {
    json.attribute("version", "1.0");
    json.attributeArray("file_info", [&] {
      for (auto [file, index] : sourceFiles)
        json.value(file.getValue());
    });
    if (hdlFileIndex)
      json.attribute("hdl_file_index", *hdlFileIndex);
  });

  json.attributeBegin("objects");
  json.rawValue(StringRef(rawObjects).drop_front().drop_back().trim());
  json.attributeEnd();

  json.objectEnd();
}

void FileEmitter::emitLoc(llvm::json::OStream &json, FileLineColLoc loc) {
  json.attribute("file", getSourceFile(loc.getFilename()));
  if (auto line = loc.getLine()) {
    json.attribute("begin_line", line);
    json.attribute("end_line", line);
  }
  if (auto col = loc.getColumn()) {
    json.attribute("begin_column", col);
    json.attribute("end_column", col);
  }
}

/// Emit the debug info for a `DIModule`.
void FileEmitter::emitModule(llvm::json::OStream &json, DIModule *module) {
  json.objectBegin();
  json.attribute("kind", "module");
  json.attribute("obj_name", module->name.getValue()); // HGL
  // TODO: This should probably be `sv.verilogName`.
  json.attribute("module_name", module->name.getValue()); // HDL
  if (module->isExtern)
    json.attribute("isExtModule", 1);
  if (auto *op = module->op) {
    findAndEmitLoc(json, "hgl_loc", op->getLoc(), false);
    findAndEmitLoc(json, "hdl_loc", op->getLoc(), true);
  }
  json.attributeArray("port_vars", [&] {
    for (auto *var : module->variables)
      emitVariable(json, var);
  });
  json.attributeArray("children", [&] {
    for (auto *instance : module->instances)
      emitInstance(json, instance);
  });
  json.objectEnd();
}

/// Emit the debug info for a `DIInstance`.
void FileEmitter::emitInstance(llvm::json::OStream &json,
                               DIInstance *instance) {
  json.objectBegin();
  json.attribute("name", instance->name.getValue());
  json.attribute("obj_name", instance->module->name.getValue()); // HGL
  // TODO: This should probably be `sv.verilogName`.
  json.attribute("module_name", instance->module->name.getValue()); // HDL
  if (auto *op = instance->op) {
    findAndEmitLoc(json, "hgl_loc", op->getLoc(), false);
    findAndEmitLoc(json, "hdl_loc", op->getLoc(), true);
  }
  json.objectEnd();
}

/// Emit the debug info for a `DIVariable`.
void FileEmitter::emitVariable(llvm::json::OStream &json,
                               DIVariable *variable) {
  json.objectBegin();
  json.attribute("var_name", variable->name.getValue());
  findAndEmitLoc(json, "hgl_loc", variable->loc, false);
  findAndEmitLoc(json, "hdl_loc", variable->loc, true);

  if (auto value = variable->value) {
    StringAttr portName;
    auto *defOp = value.getParentBlock()->getParentOp();
    auto module = dyn_cast<hw::HWModuleOp>(defOp);
    if (!module)
      module = defOp->getParentOfType<hw::HWModuleOp>();
    if (module) {
      if (auto arg = dyn_cast<BlockArgument>(value)) {
        portName = dyn_cast_or_null<StringAttr>(
            module.getInputNames()[arg.getArgNumber()]);
      } else if (auto wireOp = value.getDefiningOp<hw::WireOp>()) {
        portName = wireOp.getNameAttr();
      } else {
        for (auto &use : value.getUses()) {
          if (auto outputOp = dyn_cast<hw::OutputOp>(use.getOwner())) {
            portName = dyn_cast_or_null<StringAttr>(
                module.getOutputNames()[use.getOperandNumber()]);
            break;
          }
        }
      }
    }
    if (auto intType = dyn_cast<IntegerType>(value.getType())) {
      json.attribute("type_name", "logic");
      if (intType.getIntOrFloatBitWidth() != 1) {
        json.attributeArray("packed_range", [&] {
          json.value(intType.getIntOrFloatBitWidth() - 1);
          json.value(0);
        });
      }
    }
    if (portName) {
      json.attributeObject(
          "value", [&] { json.attribute("sig_name", portName.getValue()); });
    }
  }

  json.objectEnd();
}

//===----------------------------------------------------------------------===//
// Output Splitting
//===----------------------------------------------------------------------===//

namespace {

/// Contextual information for HGLDD emission shared across multiple HGLDD
/// files. This struct is used to determine an initial split of debug info files
/// and to distribute work.
struct Emitter {
  DebugInfo di;
  SmallVector<FileEmitter, 0> files;

  Emitter(Operation *module, StringRef directory);
};

} // namespace

Emitter::Emitter(Operation *module, StringRef directory) : di(module) {
  // Group the DI modules according to their emitted file path. Modules that
  // don't have an emitted file path annotated are collected in a separate
  // group.
  MapVector<StringAttr, FileEmitter> groups;
  for (auto [moduleName, module] : di.moduleNodes) {
    StringAttr hdlFile;
    if (module->op)
      if (auto fileLoc = findBestLocation(module->op->getLoc(), true))
        hdlFile = fileLoc.getFilename();
    groups[hdlFile].modules.push_back(module);
  }

  // Determine the output file names and move the emitters into the `files`
  // member.
  files.reserve(groups.size());
  for (auto &[hdlFile, emitter] : groups) {
    emitter.hdlFile = hdlFile;
    emitter.outputFileName = directory;
    llvm::sys::path::append(emitter.outputFileName,
                            hdlFile ? hdlFile.getValue() : "global");
    llvm::sys::path::replace_extension(emitter.outputFileName, "dd");
    files.push_back(std::move(emitter));
  }

  // Dump some information about the files to be created.
  LLVM_DEBUG({
    llvm::dbgs() << "HGLDD files:\n";
    for (auto &emitter : files) {
      llvm::dbgs() << "- " << emitter.outputFileName << " (from "
                   << emitter.hdlFile << ")\n";
      for (auto *module : emitter.modules)
        llvm::dbgs() << "  - " << module->name << "\n";
    }
  });
}

//===----------------------------------------------------------------------===//
// Emission Entry Points
//===----------------------------------------------------------------------===//

LogicalResult debug::emitHGLDD(Operation *module, StringRef directory,
                               llvm::raw_ostream &os) {
  Emitter emitter(module, directory);
  for (auto &fileEmitter : emitter.files) {
    os << "\n// ----- 8< ----- FILE \"" + fileEmitter.outputFileName +
              "\" ----- 8< -----\n\n";
    fileEmitter.emit(os);
  }
  return success();
}

LogicalResult debug::emitSplitHGLDD(Operation *module, StringRef directory) {
  Emitter emitter(module, directory);

  auto emit = [&](auto &fileEmitter) {
    // Open the output file for writing.
    std::string errorMessage;
    auto output =
        mlir::openOutputFile(fileEmitter.outputFileName, &errorMessage);
    if (!output) {
      module->emitError(errorMessage);
      return failure();
    }

    // Emit the debug information and keep the file around.
    fileEmitter.emit(output->os());
    output->keep();
    return success();
  };

  return mlir::failableParallelForEach(module->getContext(), emitter.files,
                                       emit);
}
