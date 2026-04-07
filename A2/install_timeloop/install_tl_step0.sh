
# Install system dependencies
sudo apt-get update
sudo apt install -y build-essential

# This code adapted from accelergy-timeloop-infrastructure/Makefile:
sudo DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata \
		&& sudo apt-get install -y --no-install-recommends \
						locales \
						curl \
						git \
						wget \
						python3-dev \
						python3-pip \
						python3-venv \
						scons \
						make \
						autotools-dev \
						autoconf \
						automake \
						libtool \
		&& sudo apt-get install -y --no-install-recommends \
						g++ \
						cmake

	sudo apt-get install -y --no-install-recommends \
						g++ \
						libconfig++-dev \
						libboost-dev \
						libboost-filesystem-dev \
						libboost-iostreams-dev \
						libboost-log-dev \
						libboost-serialization-dev \
						libboost-thread-dev \
						libyaml-cpp-dev \
						libncurses5-dev \
						libtinfo-dev \
						libgpm-dev \
						libgmp-dev
