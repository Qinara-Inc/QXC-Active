OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c[3];
gate majority a,b,c
{
  cx c,b;
  cx c,a;
  ccx a,b,c;
}
gate entangle a,b { h a; cx a,b; }
entangle q[0],q[1];
majority q[0],q[1],q[2];
measure q -> c;
