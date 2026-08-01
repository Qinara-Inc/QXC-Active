OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[4];
h q;
cx q[0],q[1];
measure q -> c;
