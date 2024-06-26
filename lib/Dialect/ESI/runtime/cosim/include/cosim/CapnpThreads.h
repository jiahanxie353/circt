//===- CapnpThreads.h - ESI cosim RPC ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Various classes used to implement the RPC server classes generated by
// CapnProto. Capnp C++ RPC servers are based on 'libkj' and its asynchrony
// model, which is very foreign. This is what the 'kj' namespace is along with
// alternate collections and other utility code.
//
//===----------------------------------------------------------------------===//

#ifndef COSIM_SERVER_H
#define COSIM_SERVER_H

#include "cosim/Endpoint.h"
#include "cosim/LowLevel.h"

#include <atomic>
#include <thread>

namespace kj {
class WaitScope;
} // namespace kj

namespace esi {
namespace cosim {

/// Since Capnp is not thread-safe, client and server must be run in their own
/// threads and communicate with the outside world through thread safe channels.
class CapnpCosimThread {
public:
  CapnpCosimThread();
  ~CapnpCosimThread();

  /// Stop the thread. This is a blocking call -- it will not return until the
  /// capnp thread has stopped.
  void stop();

  // Get an endpoint by its ID.
  Endpoint *getEndpoint(std::string epId);
  // Get the low level bridge.
  LowLevel *getLowLevel() { return &lowLevelBridge; }

  // Get the ESI version and compressed manifest. Returns false if the manifest
  // has yet to be loaded.
  bool getCompressedManifest(unsigned int &esiVersion,
                             std::vector<uint8_t> &manifest) {
    esiVersion = this->esiVersion;
    manifest = compressedManifest;
    return this->esiVersion >= 0;
  }

protected:
  /// Start capnp polling loop. Does not return until stop() is called. Must be
  /// called in the same thread the RPC server/client was created. 'poll' is
  /// called on each iteration of the loop.
  void loop(kj::WaitScope &waitScope, std::function<void()> poll);

  using Lock = std::lock_guard<std::mutex>;

  EndpointRegistry endpoints;
  LowLevel lowLevelBridge;

  std::thread *myThread;
  volatile bool stopSig;
  std::mutex m;

  unsigned int esiVersion = -1;
  std::vector<uint8_t> compressedManifest;
};

/// The main RpcServer. Does not implement any capnp RPC interfaces but contains
/// the capnp main RPC server. We run the capnp server in its own thread to be
/// more responsive to network traffic and so as to not slow down the
/// simulation.
class RpcServer : public CapnpCosimThread {
public:
  /// Start and stop the server thread.
  void run(uint16_t port);

  void setManifest(unsigned int esiVersion,
                   const std::vector<uint8_t> &manifest) {
    this->esiVersion = esiVersion;
    compressedManifest = manifest;
  }

  bool registerEndpoint(std::string epId, std::string fromHostTypeId,
                        std::string toHostTypeId) {
    return endpoints.registerEndpoint(epId, fromHostTypeId, toHostTypeId);
  }

private:
  /// The thread's main loop function. Exits on shutdown.
  void mainLoop(uint16_t port);
};

/// The Capnp RpcClient.
class RpcClient : public CapnpCosimThread {
  // To hide the ugly details of the capnp headers.
  struct Impl;
  friend struct Impl;

public:
  /// Start client thread.
  void run(std::string host, uint16_t port);

private:
  void mainLoop(std::string host, uint16_t port);

  /// The 'capnp' sets this to true when it is ready to go.
  std::atomic<bool> started;
};

} // namespace cosim
} // namespace esi

#endif
