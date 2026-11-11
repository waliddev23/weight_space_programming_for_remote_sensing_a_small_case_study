| Training images | Starting point | Test accuracy, mean (per seed) | Val. accuracy after epoch 1 | Verdict vs Kaiming |
|---|---|---|---|---|
| 100% (16199) | kaiming | 82.8% (82.5, 82.3, 83.5) | 56.5% |  |
| 100% (16199) | fable as written | 82.5% (80.4, 80.2, 86.9) | 55.8% | no clear difference (-0.3) |
| 100% (16199) | fable rescaled | 86.3% (86.0, 86.7, 86.3) | 59.9% | helps (+3.6) |
| 10% (1620) | kaiming | 69.4% (69.6, 69.3, 69.4) | 23.1% |  |
| 10% (1620) | fable as written | 64.3% (73.0, 60.0, 60.0) | 35.8% | no clear difference (-5.1) |
| 10% (1620) | fable rescaled | 76.2% (73.2, 77.1, 78.2) | 37.0% | helps (+6.8) |
| 1% (163) | kaiming | 63.7% (63.4, 62.6, 65.1) | 11.7% |  |
| 1% (163) | fable as written | 66.1% (65.4, 60.8, 72.0) | 32.0% | no clear difference (+2.4) |
| 1% (163) | fable rescaled | 67.5% (64.7, 68.4, 69.4) | 23.7% | helps (+3.8) |
