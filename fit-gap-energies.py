import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares


def hω(Z,N,type):
	A = N+Z
	if type == "p":
		return 41/A**(1/3) * (1-(N-Z)/(3*A))
	elif type == "n":
		return 41/A**(1/3) * (1+(N-Z)/(3*A))
	else:
		return False

def ε(qn):
	if len(qn.shape) == 1:
		qn = qn.reshape((1,len(qn)))
	fill = np.zeros(len(qn))
	fill[qn[:,2]==qn[:,1]+1/2] = qn[qn[:,2]==qn[:,1]+1/2,1]
	fill[qn[:,2]==qn[:,1]-1/2] = -qn[qn[:,2]==qn[:,1]-1/2,1]-1
	ret = np.array([(qn[:,0]+3/2),-fill,-(qn[:,1]*(qn[:,1]+1)-qn[:,0]*(qn[:,0]+3)/2)])
	if len(qn) == 1:
		ret=ret.flatten()
	return ret.T

def X_Gaps(Z,N,β,type):
	qn = getQN(7)

	E = np.zeros((qn.shape[0],3))
	for m in range(qn.shape[0]):
		E[m,:] += ε(qn[m,:])
	
	inds = np.argsort(E@β)
	qn = qn[inds]
	num = qn[:,2]*2+1
	num = np.array([sum(num[:n+1]) for n in range(len(num))])
	# print(type,num[num==126])
	# indices = np.zeros(len(Z))
	if type == "p":
		# print(Z)
		indices = np.array([list(np.argwhere(num<=_)[-1]) for _ in Z],dtype=int).flatten()
	elif type == "n":
		# print(N)
		indices = np.array([list(np.argwhere(num<=_)[-1]) for _ in N],dtype=int).flatten()

	# print(indices)
	# print(num[indices])
	return (ε(qn[indices+1,:])-ε(qn[indices,:]))

def Exp_Gaps(Z,N,β,type):
	return X_Gaps(Z,N,β,type)@β

def res(β,Z,N,type,obs):
	β = np.array([1]+list(β))
	# print(β)
	return (Exp_Gaps(Z,N,β,type)-obs)**2

def getQN(n):
	n = np.arange(n)
	l = []
	for m in n:
		λ = np.flip(np.arange(m,-1,-2).flatten())
		l += [λ]
	j = []
	for m in l:
		j += [np.array([m-0.5,m+0.5]).T]

	qn = []
	for nn in n:
		# print(j[nn].shape)
		for ll in range(len(l[nn])):
			for jj in range(j[nn].shape[1]):
				if j[nn][ll][jj] >= 0:
					qn += [[nn,l[nn][ll],j[nn][ll][jj]]]
	qn = np.array(qn)

	return qn

def main():
	# [1,κ,μ']
	β_p = np.array([1,0.06,0.04])
	β_n = np.array([1,0.06,0.02])

	data = pd.read_csv("data.csv",header=0)
	Z = np.array(data["Z"])
	N = np.array(data["N"])
	# print(data)

	qn_0_p = np.array([data["p_n_0"],data["p_l_0"],data["p_j_0"]]).T
	qn_p = np.array([data["p_n"],data["p_l"],data["p_j"]]).T
	qn_0_n = np.array([data["n_n_0"],data["n_l_0"],data["n_j_0"]]).T
	qn_n = np.array([data["n_n"],data["n_l"],data["n_j"]]).T

	hω_p = hω(Z,N,"p")
	hω_n = hω(Z,N,"n")

	gaps_p = np.array(data["P_Gap"])
	gaps_n = np.array(data["N_Gap"])
	# print(gaps_n,hω_n)
	gaps_p = gaps_p/hω_p
	gaps_n = gaps_n/hω_n

	X_p = X_Gaps(Z,N,β_p,"p")
	X_n = X_Gaps(Z,N,β_n,"n")

	print("χ^2 Before (protons):",sum(res(β_p[1:],Z,N,"p",gaps_p)))

	results_p = least_squares(res,β_p[1:],args=(Z,N,"p",gaps_p))
	results_n = least_squares(res,β_n[1:],args=(Z,N,"n",gaps_n))
	
	β_p = np.array([1]+list(results_p["x"]))
	β_n = np.array([1]+list(results_n["x"]))
	J_p = results_p["jac"]
	J_n = results_n["jac"]
	H_p = J_p.T@J_p
	H_n = J_n.T@J_n
	C_p = np.linalg.inv(H_p)
	C_n = np.linalg.inv(H_n)

	# β_p = results_p["x"]
	# β_n = np.array([1.,0.04020713,0.03865545])
	print("Proton Params:",β_p[1:])
	print("Proton Param Uncertainties:",np.sqrt(np.diag(C_p)))
	print("Neutron Params:",β_n[1:])
	print("Neutron Param Uncertainties:",np.sqrt(np.diag(C_n)))
	X_p = X_Gaps(Z,N,β_p,"p")
	X_n = X_Gaps(Z,N,β_n,"n")
	# β_p = np.linalg.inv(X_p.T@X_p)@X_p.T@gaps_p

	res_p = results_p["fun"]
	res_n = results_n["fun"]

	print("χ^2 After (protons):",sum(res_p))
	
	qn = getQN(7)

	# fig, ax = plt.subplots(figsize=(4, 10))
	fig, ax = plt.subplots(1,2)
	ax[0].set_title(r'$^{16}$O Spectrum')
	ax[0].scatter(
		[1]*len(qn),
		# np.arange(len(qn)),
		hω(8,8,"p")*(ε(qn)@β_p),
		s=9*10**3,
		marker="_",
		linewidths=1,
		zorder=3,
		)
	# ax[0].scatter(
	# 	[1]*len(qn),
	# 	# np.arange(len(qn)),
	# 	hω(8,8,"n")*(ε(qn)@β_n),
	# 	s=9*10**3,
	# 	marker="_",
	# 	linewidths=1,
	# 	zorder=3,
	# 	)
	ax[0].set_ylabel("Energy (MeV)")

	ax[1].set_title(r'$^{208}$Pb Spectrum')
	ax[1].scatter(
		[1]*len(qn),
		# np.arange(len(qn)),
		hω(82,126,"p")*(ε(qn)@β_p),
		s=9*10**3,
		marker="_",
		linewidths=1,
		zorder=3,
		label= "Proton Spectrum",
		)
	# ax[1].scatter(
	# 	[1]*len(qn),
	# 	# np.arange(len(qn)),
	# 	hω(82,126,"n")*(ε(qn)@β_n),
	# 	s=9*10**3,
	# 	marker="_",
	# 	linewidths=1,
	# 	zorder=3,
	# 	label= "Neutron Spectrum",
	# 	)
	# plt.legend()
	plt.savefig("figure.png",dpi=600)

if __name__=="__main__":
	main()