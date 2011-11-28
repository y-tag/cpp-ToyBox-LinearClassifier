#ifndef TOYBOX_SVMDCD_H
#define TOYBOX_SVMDCD_H

#include <vector>

#define WVector FVector
//#define WVector SVector

namespace toybox {

class SVector;
class FVector;

enum ModelType {
  DCD_BIN_L1 = 0, DCD_BIN_L2, MODEL_NUM
};

struct Parameter {
  ModelType model_type;
  int T;
  double C;
  bool verbose;
};

class SVMDCD {
  public:
    SVMDCD();
    ~SVMDCD();
    void Train(const std::vector<SVector> &x_vec,
               const std::vector<int> &y_vec,
               const struct Parameter &param);
    int Predict(const SVector &x, double *predict) const;
    int Save(const char *filename) const;
    int Load(const char *filename);
    int GetLabels(int *labels) const;
    int lnum() const { return lnum_; };
    int fnum() const { return fnum_; };
  private:
    int lnum_;
    int fnum_;
    int *labels_;
    std::vector<WVector*> weight_;
    SVMDCD(const SVMDCD&);
    void operator=(const SVMDCD&);
};

} // namespace toybox

#endif // TOYBOX_SVMDCD_H
