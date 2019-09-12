#include "recipes/models/utilities/convlm_serializer/Utils.h"
#include <glog/logging.h>
#include <module/module.h>
#include <fstream>
#include "common/Utils.h"

using fl::Variable;
using std::dynamic_pointer_cast;
using std::make_shared;
using std::shared_ptr;
using std::string;
using std::vector;

vector<ConvLMParamState> loadModelStates(const string& weightFile) {
  LOG(INFO) << "[ConvLMSerializer]: Reading pytorch model of the ConvLM";
  LOG_IF(FATAL, !w2l::fileExists(weightFile))
      << "Path to weight file " << weightFile << " doesn't exist";

  vector<ConvLMParamState> states;
  std::ifstream infile(weightFile);
  string line;
  while (getline(infile, line)) {
    std::stringstream ss;
    string weightName;
    int nDims;
    int64_t totalElements = 1;
    ss << line;
    ss >> weightName >> nDims;

    vector<int> shapes(nDims);
    string shape_str = "";
    for (int dim = 0; dim < nDims; dim++) {
      ss >> shapes[dim];
      totalElements *= shapes[dim];
      shape_str += std::to_string(shapes[dim]) + " ";
    }
    LOG(INFO) << "[LoadModelStates]: Reading state " << weightName
              << " with dims " << nDims << " and shape " << shape_str;
    vector<float> data(totalElements);
    for (int index = 0; index < totalElements; index++) {
      ss >> data[index];
    }

    auto parts = w2l::splitOnAnyOf(".", weightName, true);
    LOG_IF(FATAL, parts.size() < 2)
        << "Param name " << weightName
        << " should be in format {prefix.}layerName.paramName";
    vector<string> names = {w2l::join(".", parts.begin(), parts.end() - 2),
                            *(parts.end() - 2),
                            *(parts.end() - 1)};

    LOG_IF(FATAL, names.size() != 3)
        << "[LoadModelStates]: Error during parsing parameter name";

    af::dim4 dimensions(1, 1, 1, 1);
    // af has fortran-ordering (column-way)
    // revert axis before loading c-ordered matrices (row-way)
    vector<int> reordering = {0, 1, 2, 3};
    LOG_IF(FATAL, nDims > 4) << "[loadModelStates]: Layer " << weightName
                             << " has dimensions greater than 4. "
                             << "This is not supported by ArrayFire";
    for (int idx = nDims - 1; idx >= 0; idx--) {
      dimensions[nDims - 1 - idx] = shapes[idx];
      reordering[nDims - 1 - idx] = idx;
    }
    af::array weights = af::array(dimensions, data.data());
    weights = reorder(
        weights, reordering[0], reordering[1], reordering[2], reordering[3]);
    states.push_back({names[0], names[1], names[2], weights});
  }
  infile.close();

  return states;
}

void loadLayer(
    vector<ConvLMParamState>& states,
    vector<int>& layerIndices,
    shared_ptr<fl::Module> mainModule,
    shared_ptr<fl::Module> layer,
    string layerName,
    int paramIdx) {
  auto isConvLayer = [&layer]() {
    return dynamic_pointer_cast<fl::Conv2D>(layer) ||
        (dynamic_pointer_cast<fl::WeightNorm>(layer) &&
         layer->prettyString().find("Conv2D") != std::string::npos);
  };

  bool useGrad = false;
  int nParams = layer->params().size();
  int setIdx = -1;
  for (auto idx : layerIndices) {
    LOG_IF(FATAL, idx >= states.size())
        << "[LoadLayer]: states index is out of range";
    LOG(INFO) << "[LoadLayer]: load layer with param " << states[idx].paramName
              << " " << states[idx].weights.dims();
    Variable weights;
    if (states[idx].paramName == "weight") {
      setIdx++;
      if (dynamic_pointer_cast<fl::Embedding>(layer) ||
          dynamic_pointer_cast<fl::Linear>(
              layer)) { // a hack to load the embedding layer as a linear layer
        weights = Variable(states[idx].weights.T(), useGrad);
      } else {
        weights = Variable(states[idx].weights, useGrad);
      }
    } else if (states[idx].paramName == "weight_v") {
      setIdx = 0;
      if (isConvLayer()) {
        weights = reorder(Variable(states[idx].weights, useGrad), 0, 3, 1, 2);
      } else {
        weights = Variable(states[idx].weights, useGrad);
      }
    } else if (states[idx].paramName == "weight_g") {
      setIdx = 1;
      if (isConvLayer()) {
        weights = reorder(Variable(states[idx].weights, useGrad), 0, 3, 1, 2);
      } else {
        weights = Variable(states[idx].weights, useGrad);
      }
    } else if (states[idx].paramName == "bias") {
      setIdx = layer->params().size() - 1;
      if (isConvLayer()) {
        weights = reorder(Variable(states[idx].weights, useGrad), 1, 2, 0, 3);
      } else {
        weights = Variable(states[idx].weights, useGrad);
      }
    } else {
      LOG(FATAL) << "[LoadLayer]: Unknown weights param "
                 << states[idx].paramName << " for file layer "
                 << states[idx].layerName
                 << " during loading weights into the model";
    }
    LOG_IF(FATAL, setIdx >= nParams)
        << "[LoadLayer]: Incorrect index of parameter for the file layer "
        << states[idx].layerName << ". There are " << nParams
        << " parameters in the module "
        << " but you are trying to set parameter with index " << setIdx;
    LOG_IF(FATAL, weights.dims() != layer->params()[setIdx].dims())
        << "[CheckSetParams]: The state provides incorrect dimensions for weight tensor."
        << " Loading (layer " << states[idx].paramName
        << ") param dim: " << weights.dims() << " Layer (" << layerName
        << ") param dim: " << layer->params()[setIdx].dims();
    mainModule->setParams(weights, setIdx + paramIdx);
  }
}

void loadModule(
    vector<ConvLMParamState>& states,
    shared_ptr<fl::Module> mainModule,
    shared_ptr<fl::Module> subModule,
    int& loadIdx,
    int paramIdx) {
  int nParams = subModule->params().size();
  string moduleName = subModule->prettyString();
  // if no parameters for layer then skip loading weights for it
  if (nParams == 0) {
    LOG(INFO) << "[LoadModule]: Skip loading params for " << moduleName;
    return;
  }

  if (dynamic_pointer_cast<fl::Sequential>(subModule) != nullptr) {
    // in the sequential block
    LOG(INFO) << "[LoadModule]: Load sequential block " << moduleName;
    auto moduleCast = dynamic_pointer_cast<fl::Sequential>(subModule);
    auto submodules = moduleCast->modules();
    for (auto smd : submodules) {
      loadModule(states, mainModule, smd, loadIdx, paramIdx);
      paramIdx += smd->params().size();
    }
  } else if (dynamic_pointer_cast<fl::Residual>(subModule) != nullptr) {
    // in the res block
    LOG(INFO) << "[LoadModule]: Load residual block " << moduleName;
    auto moduleCast = dynamic_pointer_cast<fl::Residual>(subModule);
    auto submodules = moduleCast->modules();
    auto projectionIndices = moduleCast->getProjectionsIndices();
    std::vector<int64_t> cumParamSize(submodules.size());
    for (int ind = 0; ind < submodules.size(); ind++) {
      if (ind > 0) {
        cumParamSize[ind] =
            cumParamSize[ind - 1] + submodules[ind - 1]->params().size();
      }
      // load modules before loading projection matrices
      if (projectionIndices.find(ind) == projectionIndices.end()) {
        loadModule(
            states,
            mainModule,
            submodules[ind],
            loadIdx,
            paramIdx + cumParamSize[ind]);
      }
    }
    for (int ind = 0; ind < submodules.size(); ind++) {
      if (projectionIndices.find(ind) != projectionIndices.end()) {
        loadModule(
            states,
            mainModule,
            submodules[ind],
            loadIdx,
            paramIdx + cumParamSize[ind]);
      }
    }
  } else if (dynamic_pointer_cast<fl::AdaptiveSoftMaxLoss>(subModule)) {
    LOG(INFO) << "[LoadModule]: Load adaptive softmax loss " << moduleName;
    vector<int> moduleStateIndices(subModule->params().size());
    std::iota(moduleStateIndices.begin(), moduleStateIndices.end(), loadIdx);
    loadIdx += subModule->params().size();
    loadLayer(
        states,
        moduleStateIndices,
        mainModule,
        subModule,
        moduleName,
        paramIdx);
  } else {
    // collect indices for all weights corresponding to the same layer name
    LOG_IF(FATAL, loadIdx >= states.size())
        << "[LoadModule]: states index is out of range";
    string loadModuleName = states[loadIdx].layerName;
    vector<int> moduleStateIndices({loadIdx++});
    while ((loadIdx < states.size()) &&
           (states[loadIdx].layerName == loadModuleName)) {
      moduleStateIndices.push_back(loadIdx);
      loadIdx++;
    }
    LOG(INFO) << "[LoadModule]: Load module " << loadModuleName << " into "
              << moduleName;
    loadLayer(
        states,
        moduleStateIndices,
        mainModule,
        subModule,
        moduleName,
        paramIdx);
  }
}

void setParams(
    shared_ptr<fl::Module> network,
    shared_ptr<fl::BinaryModule> criterion,
    vector<ConvLMParamState>& states) {
  LOG(INFO) << "[SetParams]: Load weights into the model";

  int loadIdx = 0, paramIdx = 0;
  auto networkCast = dynamic_pointer_cast<fl::Sequential>(network);
  for (auto module : networkCast->modules()) {
    loadModule(states, networkCast, module, loadIdx, paramIdx);
    paramIdx += module->params().size();
  }
  if ((criterion != nullptr) && (criterion->params().size() > 0)) {
    loadModule(states, criterion, criterion, loadIdx, 0);
  }
  LOG_IF(FATAL, loadIdx < states.size())
      << "[SetParams]: Some weights are remain in the file during loading the model";
  LOG(INFO)
      << "[SetParams]: Finish loading weight from the file into the model";
}

void loadConvLM(
    shared_ptr<fl::Module>& network,
    shared_ptr<fl::BinaryModule>& criterion,
    const string& archFile,
    const string& weightFile,
    int outputTokensDim,
    const vector<int>& adaptiveTail /*  = std::vector<int>() */,
    int inputSizeAdaptiveSoftmax /* = 0 */) {
  LOG_IF(FATAL, !w2l::fileExists(archFile))
      << "Path to arch file " << archFile << " doesn't exist";
  LOG_IF(FATAL, !w2l::fileExists(weightFile))
      << "Path to weight file " << weightFile << " doesn't exist";
  // create network and criterion
  network = w2l::createW2lSeqModule(archFile, 1, outputTokensDim);
  network->eval();

  if (adaptiveTail.size() > 0) {
    auto activation = make_shared<fl::AdaptiveSoftMax>(
        inputSizeAdaptiveSoftmax, adaptiveTail);
    criterion = make_shared<fl::AdaptiveSoftMaxLoss>(activation);
    criterion->eval();
  } else {
    criterion = nullptr;
  }

  // Loading weights from the binary file
  LOG(INFO) << "[LoadConvLM]: Load states";
  auto modelStates = loadModelStates(weightFile);
  LOG_IF(
      FATAL,
      modelStates.size() !=
          network->params().size() +
              (criterion ? criterion->params().size() : 0))
      << "mismatch between the number of parameters in the arch file and the weight file "
      << modelStates.size() << " model states vs " << network->params().size()
      << " nn params + " << (criterion ? criterion->params().size() : 0)
      << " criterion params";

  // Load weight states into network and criterion
  LOG(INFO) << "[LoadConvLM]: set params";
  setParams(network, criterion, modelStates);
}
