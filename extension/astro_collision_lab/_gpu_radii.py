"""Device-only physical particle radii; imported only after explicit GPU initialization."""
import warp as wp

@wp.kernel
def _radii32(mass: wp.array(dtype=wp.float32), density: wp.array(dtype=wp.float32),
             radii: wp.array(dtype=wp.float32)):
    i=wp.tid()
    r=float(0.0)
    if mass[i]>0.0 and density[i]>0.0:
        r=wp.pow((3.0*mass[i])/(12.566370614359172*density[i]),1.0/3.0)
    radii[i]=r

@wp.kernel
def _radii64(mass: wp.array(dtype=wp.float64), density: wp.array(dtype=wp.float64),
             radii: wp.array(dtype=wp.float32)):
    i=wp.tid()
    r=wp.float64(0.0)
    if mass[i]>wp.float64(0.0) and density[i]>wp.float64(0.0):
        r=wp.pow((wp.float64(3.0)*mass[i])/(wp.float64(12.566370614359172)*density[i]),wp.float64(1.0/3.0))
    radii[i]=wp.float32(r)

class DeviceRadii:
    """One owned float32 output allocation, refreshed on the solver's current stream."""
    def __init__(self,context,count):
        self.device=context.device
        self.count=count
        self.radii=wp.empty(count,dtype=wp.float32,device=self.device)

    def update(self,channels):
        mass,density=channels['masses'],channels['densities']
        if mass.dtype!=density.dtype:
            raise TypeError('mass/density precision must agree')
        if mass.dtype not in (wp.float32,wp.float64):
            raise TypeError('radius derivation requires floating mass/density')
        kernel=_radii64 if mass.dtype==wp.float64 else _radii32
        wp.launch(kernel,dim=self.count,inputs=[mass,density],outputs=[self.radii],device=self.device)
        return self.radii
