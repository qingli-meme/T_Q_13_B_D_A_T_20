# QuRA Support-Pruning Summary

Operating point: CA >= 91.3%, ASR >= 99%.

| variant | CA | ASR | step | clean dist | trig dist | depth min | flip | params | scope | pass? |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 20_r2_all_reference | 91.34 | 99.36 | 4500 | 0.3305 | 0.8094 | 0.3151 | 0.0858 | 71606 | all | YES |
| 21_r2_all_no_conv1 | 91.28 | 99.14 | 2500 | 0.3630 | 0.8559 | -0.0043 | 0.0836 | 71424 | freeze_conv1 | NO |
| 22_r2_all_no_layer1 | 91.13 | 98.43 | 3000 | 0.3847 | 0.9326 | 0.0276 | 0.0855 | 66486 | freeze_layer1 | NO |
| 23_r2_all_no_layer2 | 90.92 | 98.23 | 4750 | 0.4238 | 0.9658 | -0.2342 | 0.0879 | 62134 | freeze_layer2 | NO |
| 24_r2_all_no_layer3 | 89.80 | 94.14 | 3000 | 0.6846 | 1.3472 | -1.7219 | 0.0890 | 52662 | freeze_layer3 | NO |
| 25_r2_all_no_layer4 | 89.91 | 89.92 | 4750 | 0.7272 | 1.4779 | -1.9196 | 0.0876 | 33718 | freeze_layer4 | NO |
| 26_r2_all_no_shortcuts | 91.40 | 98.97 | 4000 | 0.3660 | 0.8549 | 0.2212 | 0.0866 | 68918 | freeze_shortcuts | NO |
