using Pkg

# setup registries
Pkg.Registry.add("General")
Pkg.Registry.add(RegistrySpec(url = "https://github.com/wildart/BoffinStuff.git"))
Pkg.Registry.update()
