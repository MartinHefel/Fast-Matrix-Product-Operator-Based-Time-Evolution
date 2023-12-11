# Import TeNW
import TeNW

# Import Packages
import matplotlib.pyplot as plt

# Initiate Backend and Truncator
backend = TeNW.backend.initiate_backend(btype='torch', device='gpu', dtype='complex128')
truncator = TeNW.truncator.initiate_truncator(backend, truncator='QR+CBE')

# Initiate Model
L = 20
delta_t = 0.005
model = TeNW.model.ClockModel_NNN(backend, L=L, d=5, g=3, J1=1, J2=1)
model.init_MPS(key='Z')
model.TE_MPO.init_MPO(dt=delta_t)
Z = model.TE_MPO.Z

# Initiate Time Evolution Engine
engine = TeNW.algorithm.time_evolution_engine(model)

# Initiate Result Lists
res_list = [[],[],[],[],[],[]]

# Calculate Values for initial state (t=0)
res_list[0].append( 0 )
res_list[1].append( model.MPS.get_expVal_of_Op(Z, True)[L//2].real )
res_list[2].append( model.MPS.get_vN_entropy()[L//2] )
res_list[3].append( max(model.MPS.get_bonds()) )
res_list[4].append( None )
res_list[5].append( None )

# Time Evolution
N_steps = 100
options = {
            'N_sweeps': 1,
            'trunc_threshold': 1e-12,
            'chi_max': 100,
            'compute_error': True,
            'compute_distance': True,
            }
for i in range(N_steps):
    results = engine.evolve_time_step(truncator, options)
    res_list[0].append( res_list[0][-1]+delta_t )
    res_list[1].append( model.MPS.get_expVal_of_Op(Z, True)[L//2].real )
    res_list[2].append( model.MPS.get_vN_entropy()[L//2] )
    res_list[3].append( max(model.MPS.get_bonds()) )
    res_list[4].append( max(results['truncation_error']) )
    res_list[5].append( results['distance'][-1] )

# Converte/copy memory from GPU to CPU to prevent bugs
for i in range(len(res_list)):
    for j in range(len(res_list[i])):
        res_list[i][j] = backend.copy_to_np_CPU(res_list[i][j])

# Plot Results
l_list = len(res_list)-1
y_label_list = ['$\\mathrm{Re}\\langle Z^{['+str(L//2)+']}\\rangle$',
                '$S_\\mathrm{vN}$', 
                '$\\chi$', 
                '$\\epsilon_\\mathrm{trunc}$',
                '$\\Delta$']
fig, axs = plt.subplots(l_list, figsize=(5, 2*l_list))
for i in range(l_list):
    axs[i].plot(res_list[0], res_list[i+1])
    axs[i].set_ylabel(y_label_list[i])
    if i != l_list-1:
        axs[i].sharex(axs[-1])
        axs[i].tick_params('x', labelbottom=False)
axs[-2].set_yscale('log')
axs[-1].set_yscale('log')
axs[-1].set_xlabel('$t$')
plt.tight_layout()
plt.show()