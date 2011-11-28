#ifndef TOYBOX_VECTORS_H
#define TOYBOX_VECTORS_H

#include <string>
#include <vector>
#include <utility>

namespace toybox {

class FVector;
class SVector;

//typedef float VFloat;
typedef double VFloat;

class FVector {
  friend class SVector;
  public:
    FVector();
    explicit FVector(int n);
    explicit FVector(const SVector &svec);
    FVector(double *array, int n);
    explicit FVector(const std::vector<double> &vec);
    explicit FVector(const std::vector<std::pair<int, double> > &pvec);
    explicit FVector(const char *str);
    ~FVector();
    double Get(int i) const;
    void Set(int i, double v);
    void Add(const SVector &svec, double c);
    double Dot(const SVector &svec) const;
    double Dot(const FVector &fvec) const;
    void Scale(double c);
    void Show() const;
    void ToString(std::string *str) const;
    int size() const { return size_; };
    double snorm() const { return snorm_; };
  private:
    VFloat *values_;
    double scale_;
    int size_;
    double snorm_;
};

class SVector {
  friend class FVector;
  public:
    struct Pair {
      int index;
      VFloat value;
    };
    SVector();
    explicit SVector(int n);
    SVector(const SVector &svec);
    explicit SVector(const FVector &fvec);
    SVector(double *array, int n);
    explicit SVector(const std::vector<double> &vec);
    explicit SVector(const std::vector<std::pair<int, double> > &pvec);
    explicit SVector(const char *str);
    ~SVector();
    double Get(int i) const;
    void Set(int i, double v);
    void Add(const SVector &svec, double c);
    double Dot(const SVector &svec) const;
    void Scale(double c);
    void Show() const;
    void ToString(std::string *str) const;
    int size() const { return size_; };
    double snorm() const { return snorm_; };
  private:
    struct Pair *pairs_;
    int num_pairs_; // not include terminater (>= 0)
    double scale_;
    int size_;      // max index + 1
    int capacity_;  // include terminater (>= 1)
    double snorm_;
};


} // namespace toybox

#endif // TOYBOX_VECTORS_H
