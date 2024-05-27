\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  30000 y[0,0] + 30000 y[1,0] + 0 z[0,0] + 0 z[1,0] + 0 z[1,1]
Subject To
 C1: 0.5 y[0,0] + 0.5 y[1,0] >= 0.8
 C2[0]: y[0,0] = 1
 C2[1]: y[1,0] = 1
 C5[0,1]: - 300 z[0,1] >= -300
 C6[0,0]: - 300 y[0,0] >= -300
 C6[1,0]: - 300 y[1,0] >= -300
 C7a[A1]: - 0.024 y[0,0] - 0.024 y[1,0] <= 0
 C7b[A1]: - 0.024 y[0,0] - 0.024 y[1,0] <= 0
 C4[0,1]: z[0,1] + [ - y[0,0] * y[1,0] ] = 0
Bounds
Binaries
 y[0,0] y[1,0] z[0,0] z[0,1] z[1,0] z[1,1]
End
