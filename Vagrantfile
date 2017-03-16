# Vagrantfile API/syntax version. Don't touch unless you know what you're doing!
VAGRANTFILE_API_VERSION = "2"

Vagrant.configure(VAGRANTFILE_API_VERSION) do |config|

  # Every Vagrant virtual environment requires a box to build off of.
  config.vm.box = "esss/centos-7.1-desktop"

  # Create a private network, which allows host-only access to the machine
  # using a specific IP.
  #config.vm.network :private_network, ip: "192.168.59.104"
  config.vm.network :public_network, bridge: "en0: Wi-Fi (AirPort)"

  config.vm.provider "virtualbox" do |vb|
    vb.name = "ESS Centos 7.1 for Deep Learning"
    vb.customize [
      "modifyvm", :id,
      "--memory", "4096",
      "--vram", "256"
    ]
  end

  config.vm.provision :shell, path: "provisionings/libs.sh"
  config.vm.provision :shell, path: "provisionings/rbenv.sh"
  config.vm.provision :shell, path: "provisionings/pyenv.sh"
  config.vm.provision :shell, path: "provisionings/ruby2.3.1.sh"
  config.vm.provision :shell, path: "provisionings/anaconda3-4.1.1.sh"
  config.vm.provision :shell, path: "provisionings/env.sh"
  config.vm.provision :shell, path: "provisionings/os_env.sh"
  config.vm.provision :shell, path: "provisionings/gems.sh"
  config.vm.provision :shell, privileged: false, path: "provisionings/user_env.sh", run: "always"
end
