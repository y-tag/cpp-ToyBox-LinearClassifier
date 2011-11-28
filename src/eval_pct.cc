#include "perceptron.h"
#include "vectors.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>

#include <fstream>
#include <string>
#include <vector>
#include <utility>

int read_data_file(const std::string &file_name,
                   std::vector<toybox::SVector> *x_vec,
                   std::vector<int> *y_vec) { 
  std::ifstream ifs;
  std::string buff;

  if (x_vec == NULL || y_vec == NULL) {
    return 0;
  }

  std::vector<std::pair<int, double> > sv;

  ifs.open(file_name.c_str());
  while (getline(ifs, buff)) {
    char cbuff[buff.size() + 1];
    memmove(cbuff, buff.c_str(), buff.size() + 1);
    char *p = strtok(cbuff, " \t");
    y_vec->push_back(static_cast<int>(strtol(p, NULL, 10)));
    sv.clear();
    while (1) {
      char *f = strtok(NULL, ":");
      char *v = strtok(NULL, " \t");
      if (v == NULL) {
        break;
      }
      sv.push_back(std::make_pair(static_cast<int>(strtol(f, NULL, 10)),
                                  strtod(v, NULL)));
    }
    toybox::SVector x(sv);
    x_vec->push_back(x);
  }

  return 1;
}

void print_usage() {
    fprintf(stderr,
            "Usage: eval-pct train_file test_file\n"
            "options:\n"
            "-m model_type: chose model type (default 0)\n"
            "  0: 1 vs. All\n"
            "  1: Multi Class\n"
            "-t number: set number of epochs (default 10)\n"
            "-v: verbose mode\n"
            );
    return;
}


int main(int argc, char **argv) {

  struct toybox::Parameter param;

  int ch;
  extern char *optarg;
  extern int optind;

  param.model_type = toybox::BIN;
  param.T = 10;
  param.verbose = false;

  while ((ch = getopt(argc, argv, "m:t:v")) != -1) {
    switch (ch) {
      case 'm':
        param.model_type = static_cast<toybox::ModelType>(strtol(optarg, NULL, 10));
        break;
      case 't':
        param.T = static_cast<int>(strtol(optarg, NULL, 10));
        break;
      case 'v':
        param.verbose = true;
        break;
      default:
        print_usage();
        return 1;
    }
  }
  if (optind >= argc) {
    print_usage();
    return 1;
  }

  argc -= optind;
  argv += optind;

  if (argc < 2) {
    print_usage();
    return 1;
  }
  const std::string train_f = argv[0];
  const std::string test_f  = argv[1];

  toybox::Perceptron *classifier = new toybox::Perceptron();

  std::vector<toybox::SVector> x_vec;
  std::vector<int> y_vec;

  read_data_file(train_f, &x_vec, &y_vec);
  classifier->Train(x_vec, y_vec, param);

  /*
  classifier->Save("test.model");
  delete classifier; classifier = NULL;
  classifier = new toybox::Perceptron();
  classifier->Load("test.model");
  */

  x_vec.clear();
  y_vec.clear();

  int *labels = new int[classifier->lnum()];
  classifier->GetLabels(labels);

  read_data_file(test_f, &x_vec, &y_vec);
  int num_correct = 0;
  for (size_t i = 0; i < x_vec.size(); ++i) {
    int max_label = classifier->Predict(x_vec[i], NULL);
    if (labels[max_label] == y_vec[i]) {
      num_correct += 1;
    }
  }

  fprintf(stderr, "Accuracy = %.5f%% (%d/%d)\n",
          100.0 * num_correct / x_vec.size(), num_correct, x_vec.size());


  delete classifier;

  return 0;
}



