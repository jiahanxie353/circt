// RUN: circt-translate %s --emit-hgldd | FileCheck %s
// RUN: circt-translate %s --emit-split-hgldd --hgldd-output-dir=%T
// RUN: cat %T/Foo.dd | FileCheck %s --check-prefix=CHECK-FOO
// RUN: cat %T/Bar.dd | FileCheck %s --check-prefix=CHECK-BAR

#loc1 = loc("InputFoo.scala":4:10)
#loc2 = loc("InputFoo.scala":5:11)
#loc3 = loc("InputFoo.scala":6:12)
#loc4 = loc("InputBar.scala":8:5)
#loc5 = loc("InputBar.scala":14:5)
#loc6 = loc("InputBar.scala":21:10)
#loc7 = loc("InputBar.scala":22:11)
#loc8 = loc("InputBar.scala":23:12)
#loc9 = loc("InputBar.scala":25:15)
#loc10 = loc("Foo.sv":42:10)
#loc11 = loc("Bar.sv":49:10)

// CHECK-FOO: "kind": "module"
// CHECK-FOO: "obj_name": "Foo"

// CHECK-BAR: "kind": "module"
// CHECK-BAR: "obj_name": "Bar"

// CHECK-LABEL: FILE "Foo.dd"
// CHECK: {
// CHECK-NEXT: "HGLDD"
// CHECK-NEXT:   "version": "1.0"
// CHECK-NEXT:   "file_info": [
// CHECK-NEXT:     "InputFoo.scala"
// CHECK-NEXT:     "Foo.sv"
// CHECK-NEXT:     "InputBar.scala"
// CHECK-NEXT:   ]
// CHECK-NEXT:   "hdl_file_index": 2
// CHECK-NEXT: }
// CHECK-NEXT: "objects"
// CHECK-NEXT: {
// CHECK-NEXT:   "kind": "module"
// CHECK-NEXT:   "obj_name": "Foo"
// CHECK-NEXT:   "module_name": "Foo"
// CHECK-NEXT:   "hgl_loc"
// CHECK-NEXT:     "begin_column": 10
// CHECK-NEXT:     "begin_line": 4
// CHECK-NEXT:     "end_column": 10
// CHECK-NEXT:     "end_line": 4
// CHECK-NEXT:     "file": 1
// CHECK-NEXT:   }
// CHECK-NEXT:   "hdl_loc"
// CHECK-NEXT:     "begin_column": 10
// CHECK-NEXT:     "begin_line": 42
// CHECK-NEXT:     "end_column": 10
// CHECK-NEXT:     "end_line": 42
// CHECK-NEXT:     "file": 2
// CHECK-NEXT:   }
// CHECK-NEXT:   "port_vars"
// CHECK:          "var_name": "inA"
// CHECK:          "var_name": "outB"
// CHECK:        "children"
// CHECK-LABEL:    "name": "b0"
// CHECK:          "obj_name": "Bar"
// CHECK:          "module_name": "Bar"
// CHECK:          "hgl_loc"
// CHECK:            "file": 3
// CHECK-LABEL:    "name": "b1"
// CHECK:          "obj_name": "Bar"
// CHECK:          "module_name": "Bar"
// CHECK:          "hgl_loc"
// CHECK:            "file": 3
hw.module @Foo(%a: i32 loc(#loc2)) -> (b: i32 loc(#loc3)) {
  dbg.variable "inA", %a : i32 loc(#loc2)
  dbg.variable "outB", %b1.y : i32 loc(#loc3)
  %c42_i8 = hw.constant 42 : i8
  dbg.variable "var1", %c42_i8 : i8 loc(#loc3)
  %b0.y = hw.instance "b0" @Bar(x: %a: i32) -> (y: i32) loc(#loc4)
  %b1.y = hw.instance "b1" @Bar(x: %b0.y: i32) -> (y: i32) loc(#loc5)
  hw.output %b1.y : i32 loc(#loc1)
} loc(fused[#loc1, "emitted"(#loc10)])

// CHECK-LABEL: FILE "Bar.dd"
// CHECK: "module_name": "Bar"
// CHECK:   "port_vars"
// CHECK:     "var_name": "inX"
// CHECK:     "var_name": "outY"
// CHECK:     "var_name": "varZ"
hw.module private @Bar(%x: i32 loc(#loc7)) -> (y: i32 loc(#loc8)) {
  %0 = comb.mul %x, %x : i32 loc(#loc9)
  dbg.variable "inX", %x : i32 loc(#loc7)
  dbg.variable "outY", %0 : i32 loc(#loc8)
  dbg.variable "varZ", %0 : i32 loc(#loc9)
  %1 = comb.add %0, %x {hw.verilogName = "some_verilog_name"} : i32 loc(#loc9)
  dbg.variable "add", %1 : i32
  hw.output %0 : i32 loc(#loc6)
} loc(fused[#loc6, "emitted"(#loc11)])

// CHECK-LABEL: FILE "global.dd"
// CHECK-LABEL: "module_name": "Aggregates"
// CHECK:         "var_name": "data"
hw.module @Aggregates(%data_a: i32, %data_b: index, %data_c_0: i17, %data_c_1: i17) {
  %0 = dbg.array [%data_c_0, %data_c_1] : i17
  %1 = dbg.struct {"a": %data_a, "b": %data_b, "c": %0} : i32, index, !dbg.array<i17>
  dbg.variable "data", %1 : !dbg.struct<i32, index, !dbg.array<i17>>
}

// CHECK-LABEL: "module_name": "Expressions"
hw.module @Expressions(%a: i1, %b: i1) {
  // CHECK-LABEL: "var_name": "blockArg"
  // CHECK: "value": {"sig_name":"a"}
  // CHECK: "type_name": "logic"
  dbg.variable "blockArg", %a : i1

  // CHECK-LABEL: "var_name": "explicitlyNamed"
  // CHECK: "value": {"sig_name":"explicitName"}
  // CHECK: "type_name": "logic"
  %0 = comb.add %a, %a {hw.verilogName = "explicitName"} : i1
  dbg.variable "explicitlyNamed", %0 : i1

  // CHECK-LABEL: "var_name": "instPort"
  // CHECK: "value": {"field":"outPort","var_ref":{"sig_name":"someInst"}}
  // CHECK: "type_name": "logic"
  %1 = hw.instance "someInst" @SingleResult() -> (outPort: i1)
  dbg.variable "instPort", %1 : i1

  // CHECK-LABEL: "var_name": "const"
  // CHECK: "value": {"constant":"9001"}
  // CHECK: "type_name": "logic"
  // CHECK: "packed_range": [
  // CHECK-NEXT: 41
  // CHECK-NEXT: 0
  // CHECK-NEXT: ]
  %2 = hw.constant 9001 : i42
  dbg.variable "const", %2 : i42

  // CHECK-LABEL: "var_name": "readWire"
  // CHECK: "value": {"sig_name":"svWire"}
  // CHECK: "type_name": "logic"
  %svWire = sv.wire : !hw.inout<i1>
  %3 = sv.read_inout %svWire : !hw.inout<i1>
  dbg.variable "readWire", %3 : i1

  // CHECK-LABEL: "var_name": "readReg"
  // CHECK: "value": {"sig_name":"svReg"}
  // CHECK: "type_name": "logic"
  %svReg = sv.reg : !hw.inout<i1>
  %4 = sv.read_inout %svReg : !hw.inout<i1>
  dbg.variable "readReg", %4 : i1

  // CHECK-LABEL: "var_name": "readLogic"
  // CHECK: "value": {"sig_name":"svLogic"}
  // CHECK: "type_name": "logic"
  %svLogic = sv.logic : !hw.inout<i1>
  %5 = sv.read_inout %svLogic : !hw.inout<i1>
  dbg.variable "readLogic", %5 : i1

  // CHECK-LABEL: "var_name": "wire"
  // CHECK: "value": {"sig_name":"hwWire"}
  // CHECK: "type_name": "logic"
  %hwWire = hw.wire %a : i1
  dbg.variable "wire", %hwWire : i1

  // CHECK-LABEL: "var_name": "unaryParity"
  // CHECK: "value": {"opcode":"^","operands":[{"sig_name":"a"}]}
  // CHECK: "type_name": "logic"
  %6 = comb.parity %a : i1
  dbg.variable "unaryParity", %6 : i1

  // CHECK-LABEL: "var_name": "binaryAdd"
  // CHECK: "value": {"opcode":"+","operands":[{"sig_name":"a"},{"sig_name":"b"}]}
  // CHECK: "type_name": "logic"
  %7 = comb.add %a, %b : i1
  dbg.variable "binaryAdd", %7 : i1

  // CHECK-LABEL: "var_name": "binarySub"
  // CHECK: "value": {"opcode":"-","operands":[{"sig_name":"a"},{"sig_name":"b"}]}
  // CHECK: "type_name": "logic"
  %8 = comb.sub %a, %b : i1
  dbg.variable "binarySub", %8 : i1

  // CHECK-LABEL: "var_name": "binaryMul"
  // CHECK: "value": {"opcode":"*","operands":[{"sig_name":"a"},{"sig_name":"b"}]}
  // CHECK: "type_name": "logic"
  %9 = comb.mul %a, %b : i1
  dbg.variable "binaryMul", %9 : i1

  // CHECK-LABEL: "var_name": "binaryDiv1"
  // CHECK: "value": {"opcode":"/","operands":[{"sig_name":"a"},{"sig_name":"b"}]}
  // CHECK: "type_name": "logic"
  // CHECK-LABEL: "var_name": "binaryDiv2"
  // CHECK: "value": {"opcode":"/","operands":[{"sig_name":"a"},{"sig_name":"b"}]}
  // CHECK: "type_name": "logic"
  %10 = comb.divu %a, %b : i1
  %11 = comb.divs %a, %b : i1
  dbg.variable "binaryDiv1", %10 : i1
  dbg.variable "binaryDiv2", %11 : i1

  // CHECK-LABEL: "var_name": "binaryMod1"
  // CHECK: "value": {"opcode":"%","operands":[{"sig_name":"a"},{"sig_name":"b"}]}
  // CHECK: "type_name": "logic"
  // CHECK-LABEL: "var_name": "binaryMod2"
  // CHECK: "value": {"opcode":"%","operands":[{"sig_name":"a"},{"sig_name":"b"}]}
  // CHECK: "type_name": "logic"
  %12 = comb.modu %a, %b : i1
  %13 = comb.mods %a, %b : i1
  dbg.variable "binaryMod1", %12 : i1
  dbg.variable "binaryMod2", %13 : i1

  // CHECK-LABEL: "var_name": "binaryShl"
  // CHECK: "value": {"opcode":"<<","operands":[{"sig_name":"a"},{"sig_name":"b"}]}
  // CHECK: "type_name": "logic"
  // CHECK-LABEL: "var_name": "binaryShr1"
  // CHECK: "value": {"opcode":">>","operands":[{"sig_name":"a"},{"sig_name":"b"}]}
  // CHECK: "type_name": "logic"
  // CHECK-LABEL: "var_name": "binaryShr2"
  // CHECK: "value": {"opcode":">>>","operands":[{"sig_name":"a"},{"sig_name":"b"}]}
  // CHECK: "type_name": "logic"
  %14 = comb.shl %a, %b : i1
  %15 = comb.shru %a, %b : i1
  %16 = comb.shrs %a, %b : i1
  dbg.variable "binaryShl", %14 : i1
  dbg.variable "binaryShr1", %15 : i1
  dbg.variable "binaryShr2", %16 : i1

  // CHECK-LABEL: "var_name": "cmpEq"
  // CHECK: "value": {"opcode":"==","operands":[{"sig_name":"a"},{"sig_name":"b"}]}
  // CHECK: "type_name": "logic"
  // CHECK-LABEL: "var_name": "cmpNe"
  // CHECK: "value": {"opcode":"!=","operands":[{"sig_name":"a"},{"sig_name":"b"}]}
  // CHECK: "type_name": "logic"
  // CHECK-LABEL: "var_name": "cmpCeq"
  // CHECK: "value": {"opcode":"===","operands":[{"sig_name":"a"},{"sig_name":"b"}]}
  // CHECK: "type_name": "logic"
  // CHECK-LABEL: "var_name": "cmpCne"
  // CHECK: "value": {"opcode":"!==","operands":[{"sig_name":"a"},{"sig_name":"b"}]}
  // CHECK: "type_name": "logic"
  // CHECK-LABEL: "var_name": "cmpWeq"
  // CHECK: "value": {"opcode":"==?","operands":[{"sig_name":"a"},{"sig_name":"b"}]}
  // CHECK: "type_name": "logic"
  // CHECK-LABEL: "var_name": "cmpWne"
  // CHECK: "value": {"opcode":"!=?","operands":[{"sig_name":"a"},{"sig_name":"b"}]}
  // CHECK: "type_name": "logic"
  // CHECK-LABEL: "var_name": "cmpUlt"
  // CHECK: "value": {"opcode":"<","operands":[{"sig_name":"a"},{"sig_name":"b"}]}
  // CHECK: "type_name": "logic"
  // CHECK-LABEL: "var_name": "cmpSlt"
  // CHECK: "value": {"opcode":"<","operands":[{"sig_name":"a"},{"sig_name":"b"}]}
  // CHECK: "type_name": "logic"
  // CHECK-LABEL: "var_name": "cmpUgt"
  // CHECK: "value": {"opcode":">","operands":[{"sig_name":"a"},{"sig_name":"b"}]}
  // CHECK: "type_name": "logic"
  // CHECK-LABEL: "var_name": "cmpSgt"
  // CHECK: "value": {"opcode":">","operands":[{"sig_name":"a"},{"sig_name":"b"}]}
  // CHECK: "type_name": "logic"
  // CHECK-LABEL: "var_name": "cmpUle"
  // CHECK: "value": {"opcode":"<=","operands":[{"sig_name":"a"},{"sig_name":"b"}]}
  // CHECK: "type_name": "logic"
  // CHECK-LABEL: "var_name": "cmpSle"
  // CHECK: "value": {"opcode":"<=","operands":[{"sig_name":"a"},{"sig_name":"b"}]}
  // CHECK: "type_name": "logic"
  // CHECK-LABEL: "var_name": "cmpUge"
  // CHECK: "value": {"opcode":">=","operands":[{"sig_name":"a"},{"sig_name":"b"}]}
  // CHECK: "type_name": "logic"
  // CHECK-LABEL: "var_name": "cmpSge"
  // CHECK: "value": {"opcode":">=","operands":[{"sig_name":"a"},{"sig_name":"b"}]}
  // CHECK: "type_name": "logic"
  %17 = comb.icmp eq %a, %b : i1
  %18 = comb.icmp ne %a, %b : i1
  %19 = comb.icmp ceq %a, %b : i1
  %20 = comb.icmp cne %a, %b : i1
  %21 = comb.icmp weq %a, %b : i1
  %22 = comb.icmp wne %a, %b : i1
  %23 = comb.icmp ult %a, %b : i1
  %24 = comb.icmp slt %a, %b : i1
  %25 = comb.icmp ugt %a, %b : i1
  %26 = comb.icmp sgt %a, %b : i1
  %27 = comb.icmp ule %a, %b : i1
  %28 = comb.icmp sle %a, %b : i1
  %29 = comb.icmp uge %a, %b : i1
  %30 = comb.icmp sge %a, %b : i1
  dbg.variable "cmpEq", %17 : i1
  dbg.variable "cmpNe", %18 : i1
  dbg.variable "cmpCeq", %19 : i1
  dbg.variable "cmpCne", %20 : i1
  dbg.variable "cmpWeq", %21 : i1
  dbg.variable "cmpWne", %22 : i1
  dbg.variable "cmpUlt", %23 : i1
  dbg.variable "cmpSlt", %24 : i1
  dbg.variable "cmpUgt", %25 : i1
  dbg.variable "cmpSgt", %26 : i1
  dbg.variable "cmpUle", %27 : i1
  dbg.variable "cmpSle", %28 : i1
  dbg.variable "cmpUge", %29 : i1
  dbg.variable "cmpSge", %30 : i1

  // CHECK-LABEL: "var_name": "variadicAnd"
  // CHECK: "value": {"opcode":"&","operands":[{"opcode":"&","operands":[{"sig_name":"a"},{"sig_name":"b"}]},{"sig_name":"explicitName"}]}
  // CHECK: "type_name": "logic"
  %31 = comb.and %a, %b, %0 : i1
  dbg.variable "variadicAnd", %31 : i1

  // CHECK-LABEL: "var_name": "variadicOr"
  // CHECK: "value": {"opcode":"|","operands":[{"opcode":"|","operands":[{"sig_name":"a"},{"sig_name":"b"}]},{"sig_name":"explicitName"}]}
  // CHECK: "type_name": "logic"
  %32 = comb.or %a, %b, %0 : i1
  dbg.variable "variadicOr", %32 : i1

  // CHECK-LABEL: "var_name": "variadicXor"
  // CHECK: "value": {"opcode":"^","operands":[{"opcode":"^","operands":[{"sig_name":"a"},{"sig_name":"b"}]},{"sig_name":"explicitName"}]}
  // CHECK: "type_name": "logic"
  %33 = comb.xor %a, %b, %0 : i1
  dbg.variable "variadicXor", %33 : i1

  // CHECK-LABEL: "var_name": "concat"
  // CHECK: "value": {"opcode":"concat","operands":[{"sig_name":"a"},{"sig_name":"b"},{"sig_name":"explicitName"}]}
  // CHECK: "type_name": "logic"
  // CHECK: "packed_range": [
  // CHECK-NEXT: 2
  // CHECK-NEXT: 0
  // CHECK-NEXT: ]
  %34 = comb.concat %a, %b, %0 : i1, i1, i1
  dbg.variable "concat", %34 : i3

  // CHECK-LABEL: "var_name": "replicate"
  // CHECK: "value": {"opcode":"repeat","operands":[{"integer_num":3},{"sig_name":"a"}]}
  // CHECK: "type_name": "logic"
  // CHECK: "packed_range": [
  // CHECK-NEXT: 2
  // CHECK-NEXT: 0
  // CHECK-NEXT: ]
  %35 = comb.replicate %a : (i1) -> i3
  dbg.variable "replicate", %35 : i3

  // CHECK-LABEL: "var_name": "extract"
  // CHECK: "value": {"opcode":"part_select","operands":[{"sig_name":"wideWire"},{"integer_num":19},{"integer_num":12}]}
  // CHECK: "type_name": "logic"
  // CHECK: "packed_range": [
  // CHECK-NEXT: 7
  // CHECK-NEXT: 0
  // CHECK-NEXT: ]
  %wideWire = hw.wire %2 : i42
  %36 = comb.extract %wideWire from 12 : (i42) -> i8
  dbg.variable "extract", %36 : i8

  // CHECK-LABEL: "var_name": "mux"
  // CHECK: "value": {"opcode":"?:","operands":[{"sig_name":"a"},{"sig_name":"b"},{"sig_name":"explicitName"}]}
  // CHECK: "type_name": "logic"
  %37 = comb.mux %a, %b, %0 : i1
  dbg.variable "mux", %37 : i1
}
hw.module.extern @SingleResult() -> (outPort: i1)
