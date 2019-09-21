#include "HDF5Data.h"

#include <iostream>
#include <memory>
#include "H5Cpp.h"


std::vector<float> loadData(const std::string& filename) {
  H5::Exception::dontPrint();  // throw exceptions when they happen

  H5::H5File file(filename, H5F_ACC_RDONLY);
  H5::DataSet dataset = file.openDataSet("features");

  // ensure data type is float
  if (dataset.getTypeClass() != H5T_FLOAT) {
    throw std::runtime_error("loadData: dataset must be of type H5T_FLOAT");
  }

  // ensure data is single dimensional, because only that is handled
  const auto dims = dataset.getSpace().getSimpleExtentNdims();
  if (dims != 1) {
    throw std::runtime_error("loadData: dataset must be single dimensional");
  }

  // get the size of the dimension
  auto dimSizeBuffer = std::unique_ptr<hsize_t[]>(new hsize_t[dims]);
  dataset.getSpace().getSimpleExtentDims(dimSizeBuffer.get(), NULL);

  // allocate memory for the buffer, and read the data
  auto dataBuffer = std::unique_ptr<float[]>(new float[dimSizeBuffer[0]]);
  dataset.read(dataBuffer.get(), H5::PredType::NATIVE_FLOAT);

  file.close();

  // initialize the vector with the array data
  return std::vector<float>(dataBuffer.get(), dataBuffer.get() + dimSizeBuffer[0]);
}
