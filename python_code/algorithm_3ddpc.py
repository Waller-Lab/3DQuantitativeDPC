import numpy as np
import time
pi    = np.pi
naxis = np.newaxis
F_2D  = lambda x: np.fft.fft2(x, axes=(0, 1))
IF_2D = lambda x: np.fft.ifft2(x, axes=(0, 1))
F_3D  = lambda x: np.fft.fftn(x, axes=(0, 1, 2))
IF_3D = lambda x: np.fft.ifftn(x, axes=(0, 1, 2))

def pupilGen(fxlin, fylin, wavelength, na, na_in=0.0):
	'''
	pupilGen create a circular pupil function in Fourier space.
	Inputs:
			fxlin     : 1D spatial frequency coordinate in horizontal direction
			fylin     : 1D spatial frequency coordinate in vertical direction
			wavelength: wavelength of incident light
			na        : numerical aperture of the imaging system
			na_in     : put a non-zero number smaller than na to generate an annular function
	Output:
			pupil     : pupil function
	'''
	pupil = np.array(fxlin[naxis, :]**2+fylin[:, naxis]**2 <= (na/wavelength)**2, dtype="float32")
	if na_in != 0.0:
		pupil[fxlin[naxis, :]**2+fylin[:, naxis]**2 < (na_in/wavelength)**2] = 0.0
	return pupil

def _genGrid(size, dx):
	'''
	_genGrid create a 1D coordinate vector.
	Inputs:
			size : length of the coordinate vector
			dx   : step size of the 1D coordinate
	Output:
			grid : 1D coordinate vector
	'''
	xlin = np.arange(size, dtype='complex64')
	return (xlin-size//2)*dx

class Solver3DDPC:
	'''
	Solver3DDPC class provides methods to preprocess 3D DPC measurements and recovers 3D refractive index with Tikhonov or TV regularziation.
	'''
	def __init__(self, dpc_imgs, wavelength, na, na_in, pixel_size, pixel_size_z, rotation, RI_medium):
		'''
		Initialize system parameters and functions for DPC phase microscopy.
		'''
		self.wavelength     = wavelength
		self.na             = na
		self.na_in          = na_in
		self.pixel_size     = pixel_size
		self.pixel_size_z   = pixel_size_z
		self.rotation       = rotation
		self.dpc_num        = len(rotation)
		self.fxlin          = np.fft.ifftshift(_genGrid(dpc_imgs.shape[1], 1.0/dpc_imgs.shape[1]/self.pixel_size))
		self.fylin          = np.fft.ifftshift(_genGrid(dpc_imgs.shape[0], 1.0/dpc_imgs.shape[0]/self.pixel_size))
		self.dpc_imgs       = dpc_imgs.astype('float32')
		self.RI_medium      = RI_medium
		self.window         = np.fft.ifftshift(np.hamming(dpc_imgs.shape[2]))
		self.pupil          = pupilGen(self.fxlin, self.fylin, self.wavelength, self.na)
		self.phase_defocus  = self.pupil*2.0*pi*((1.0/wavelength)**2-self.fxlin[naxis, :]**2-self.fylin[:, naxis]**2)**0.5
		self.oblique_factor = self.pupil/4.0/pi/((RI_medium/wavelength)**2-self.fxlin[naxis, :]**2-self.fylin[:, naxis]**2)**0.5
		self.normalization()
		self.sourceGen()
		self.WOTFGen()

	def normalization(self):
		'''
		Normalize the 3D intensity stacks by their average illumination intensities, and subtract the mean.
		'''
		self.dpc_imgs  /= np.mean(self.dpc_imgs, axis=(0, 1, 2), keepdims=True)
		self.dpc_imgs  -= 1.0

	def sourceGen(self):
		'''
		Generate DPC source patterns based on the rotation angles and numerical aperture of the illuminations.
		'''
		self.source = []
		pupil       = pupilGen(self.fxlin, self.fylin, self.wavelength, self.na, na_in=self.na_in)
		for rot_index in range(self.dpc_num):
			self.source.append(np.zeros((self.dpc_imgs.shape[:2]), dtype='float32'))
			rotdegree = self.rotation[rot_index]
			if rotdegree < 180:
				self.source[-1][self.fylin[:, naxis]*np.cos(np.deg2rad(rotdegree))+1e-15>=
				                self.fxlin[naxis, :]*np.sin(np.deg2rad(rotdegree))] = 1.0
				self.source[-1] *= pupil
			else:
				self.source[-1][self.fylin[:, naxis]*np.cos(np.deg2rad(rotdegree))+1e-15<
				                self.fxlin[naxis, :]*np.sin(np.deg2rad(rotdegree))] = -1.0
				self.source[-1] *= pupil
				self.source[-1] += pupil
		self.source = np.asarray(self.source)

	def sourceFlip(self, source):
		'''
		Flip the sources in vertical and horizontal directions, since the coordinates of the source plane and the pupil plane are opposite.
		'''
		source_flip = np.fft.fftshift(source)
		source_flip = source_flip[::-1, ::-1]
		if np.mod(source_flip.shape[0], 2)==0:
			source_flip = np.roll(source_flip, 1, axis=0)
		if np.mod(source_flip.shape[1], 2)==0:
			source_flip = np.roll(source_flip, 1, axis=1)

		return np.fft.ifftshift(source_flip)

	def WOTFGen(self):
		'''
		Generate the absorption (imaginary part) and phase (real part) weak object transfer functions (WOTFs) using the sources and the pupil.
		'''
		dim_x       = self.dpc_imgs.shape[1]
		dim_y       = self.dpc_imgs.shape[0]
		dfx         = 1.0/dim_x/self.pixel_size
		dfy         = 1.0/dim_y/self.pixel_size
		z_lin       = np.fft.ifftshift(_genGrid(self.dpc_imgs.shape[2], self.pixel_size_z))
		prop_kernel = np.exp(1.0j*z_lin[naxis, naxis, :]*self.phase_defocus[:, :, naxis])
		self.H_real = []
		self.H_imag = []
		for rot_index in range(self.dpc_num):
			source_flip      = self.sourceFlip(self.source[rot_index])
			FSP_cFPG         = F_2D(source_flip[:, :, naxis]*self.pupil[:, :, naxis]*prop_kernel)*\
						       F_2D(self.pupil[:, :, naxis]*prop_kernel*self.oblique_factor[:, :, naxis]).conj()
			self.H_real.append(2.0*IF_2D(1.0j*FSP_cFPG.imag*dfx*dfy))
			self.H_real[-1] *= self.window[naxis, naxis, :]
			self.H_real[-1]  = np.fft.fft(self.H_real[-1], axis=2)*self.pixel_size_z
			self.H_imag.append(2.0*IF_2D(FSP_cFPG.real*dfx*dfy))
			self.H_imag[-1] *= self.window[naxis, naxis, :]
			self.H_imag[-1]  = np.fft.fft(self.H_imag[-1], axis=2)*self.pixel_size_z
			total_source     = np.sum(source_flip*self.pupil*self.pupil.conj())*dfx*dfy
			self.H_real[-1] *= 1.0j/total_source
			self.H_imag[-1] *= 1.0/total_source
			print("3D weak object transfer function {:02d}/{:02d} has been evaluated.".format(rot_index+1, self.dpc_num), end="\r")
		self.H_real = np.array(self.H_real).astype('complex64')
		self.H_imag = np.array(self.H_imag).astype('complex64')

	def _V2RI(self, V_real, V_imag):
		'''
		Convert the complex scattering potential (V) into the refractive index. Imaginary part of the refractive index is dumped.
		'''
		wavenumber  = 2.0*pi/self.wavelength
		B           = -1.0*(self.RI_medium**2-V_real/wavenumber**2)
		C           = -1.0*(-1.0*V_imag/wavenumber**2/2.0)**2
		RI_obj      = ((-1.0*B+(B**2-4.0*C)**0.5)/2.0)**0.5

		return np.array(RI_obj)

	def setRegularizationParameters(self, reg_real=5e-5, reg_imag=5e-5, tau=5e-5, rho = 5e-5):
		'''
		Set regularization parameters for Tikhonov deconvolution and total variation regularization.
		'''
		# Tikhonov regularization parameters
		self.reg_real = reg_real
		self.reg_imag = reg_imag

		# TV regularization parameters
		self.tau      = tau

		# ADMM penalty parameter
		self.rho      = rho

	def _prox_LASSO(self, V1_k, y_DV_k, use_gpu):
		'''
		_prox_LASSO performs the proximal operator and solves the LASSO problem with L1 norm for total variation regularization.
		Inputs:
			V1_k      		: complex scattering potential
			y_DV_k    		: Lagrange multipliers for the gradient vectors of the scattering potential
			use_gpu   		: flag to specify gpu usage
		Output:
			DV_k    		: soft-thresholded gradient vectors of the scattering potential
			DV_k_or_diff    : difference between the thresholded gradient vectors and the original ones
		'''
		if use_gpu:
			shape_3d     = self.dpc_imgs.shape[:3]
			DV_k_or_diff = af.constant(0.0, shape_3d[0], shape_3d[1], shape_3d[2], 6, dtype=af.Dtype.f32)
			DV_k_or_diff[:,:,:,0] = V1_k[:,:,:,0] - af.shift(V1_k[:,:,:,0], 0, -1)
			DV_k_or_diff[:,:,:,1] = V1_k[:,:,:,0] - af.shift(V1_k[:,:,:,0], -1)
			DV_k_or_diff[:,:,:,2] = V1_k[:,:,:,0] - af.shift(V1_k[:,:,:,0], 0, 0, -1)
			DV_k_or_diff[:,:,:,3] = V1_k[:,:,:,1] - af.shift(V1_k[:,:,:,1], 0, -1)
			DV_k_or_diff[:,:,:,4] = V1_k[:,:,:,1] - af.shift(V1_k[:,:,:,1], -1)
			DV_k_or_diff[:,:,:,5] = V1_k[:,:,:,1] - af.shift(V1_k[:,:,:,1], 0, 0, -1)
		else:
			DV_k_or_diff = np.zeros(self.dpc_imgs.shape[:3]+ (6, ), dtype='float32')
			DV_k_or_diff[:,:,:,0] = V1_k[:,:,:,0] - np.roll(V1_k[:,:,:,0], -1, axis=1)
			DV_k_or_diff[:,:,:,1] = V1_k[:,:,:,0] - np.roll(V1_k[:,:,:,0], -1, axis=0)
			DV_k_or_diff[:,:,:,2] = V1_k[:,:,:,0] - np.roll(V1_k[:,:,:,0], -1, axis=2)
			DV_k_or_diff[:,:,:,3] = V1_k[:,:,:,1] - np.roll(V1_k[:,:,:,1], -1, axis=1)
			DV_k_or_diff[:,:,:,4] = V1_k[:,:,:,1] - np.roll(V1_k[:,:,:,1], -1, axis=0)
			DV_k_or_diff[:,:,:,5] = V1_k[:,:,:,1] - np.roll(V1_k[:,:,:,1], -1, axis=2)

		DV_k         = DV_k_or_diff - y_DV_k
		if use_gpu:
			DV_k = af.maxof(DV_k-self.tau/self.rho, 0.0) - af.maxof(-DV_k-self.tau/self.rho, 0.0)
		else:
			DV_k = np.maximum(DV_k-self.tau/self.rho, 0.0) - np.maximum(-DV_k-self.tau/self.rho, 0.0)

		DV_k_or_diff = DV_k - DV_k_or_diff

		return DV_k, DV_k_or_diff

	def _prox_projection(self, V1_k, V2_k, y_V2_k, boundary_constraint):
		'''
		_prox_projection performs Euclidean norm projection to impose positivity or negativity constraints on the scattering potential.
		Inputs:
		   V1_k                : complex scattering potential
		   V2_k                : splitted complex scattering potential
		   y_V2_k              : Lagrange multipliers for the splitted scattering potential
		   boundary_constraint : indicate whether to use positive or negative constraint on the scattering potential
		Output:
		   V2_k                : updated splitted complex scattering potential
		'''
		V2_k   = V1_k + y_V2_k
		V_real = V2_k[:,:,:,1]
		V_imag = V2_k[:,:,:,0]

		if boundary_constraint["real"]=="positive":
			V_real[V_real<0.0] = 0.0
		elif boundary_constraint["real"]=="negative":
			V_real[V_real>0.0] = 0.0

		if boundary_constraint["imag"]=="positive":
			V_imag[V_real<0.0] = 0.0
		elif boundary_constraint["imag"]=="negative":
			V_imag[V_real>0.0] = 0.0

		V2_k[:,:,:,0] = V_imag
		V2_k[:,:,:,1] = V_real

		return V2_k

	def _deconvTikhonov(self, AHA, AHy, determinant, use_gpu):
		'''
		_deconvTikhonov solves a Least-Squares problem with L2 regularization.
		'''
		if use_gpu:
			V_real      = af.real(af.ifft3((AHA[0]*AHy[1]-AHA[2]*AHy[0])/determinant))
			V_imag      = af.real(af.ifft3((AHA[3]*AHy[0]-AHA[1]*AHy[1])/determinant))
		else:
			V_real      = IF_3D((AHA[0]*AHy[1]-AHA[2]*AHy[0])/determinant).real
			V_imag      = IF_3D((AHA[3]*AHy[0]-AHA[1]*AHy[1])/determinant).real

		return V_real, V_imag

	def _deconvTV(self, AHA, determinant, fIntensity, fDx, fDy, fDz, tv_max_iter, boundary_constraint, use_gpu):
		'''
		_deconvTV solves the 3D DPC deconvolution with total variation regularization and boundary value constraints using the ADMM algorithm.
		'''
		AHy         =[(self.H_imag.conj()*fIntensity).sum(axis=0), (self.H_real.conj()*fIntensity).sum(axis=0)]

		if use_gpu:
			shape_3d    = self.dpc_imgs.shape[:3]
			V1_k        = af.constant(0.0, shape_3d[0], shape_3d[1], shape_3d[2], 2, dtype=af.Dtype.f32)
			V2_k        = af.constant(0.0, shape_3d[0], shape_3d[1], shape_3d[2], 2, dtype=af.Dtype.f32)
			DV_k        = af.constant(0.0, shape_3d[0], shape_3d[1], shape_3d[2], 6, dtype=af.Dtype.f32)
			y_DV_k      = af.constant(0.0, shape_3d[0], shape_3d[1], shape_3d[2], 6, dtype=af.Dtype.f32)
			y_V2_k      = af.constant(0.0, shape_3d[0], shape_3d[1], shape_3d[2], 2, dtype=af.Dtype.f32)
			AHy         = [af.to_array(AHy_i) for AHy_i in AHy]
		else:
			V1_k        = np.zeros(self.dpc_imgs.shape[:3]+ (2, ), dtype='float32')
			V2_k        = np.zeros(self.dpc_imgs.shape[:3]+ (2, ), dtype='float32')
			DV_k        = np.zeros(self.dpc_imgs.shape[:3]+ (6, ), dtype='float32')
			y_DV_k      = np.zeros(self.dpc_imgs.shape[:3]+ (6, ), dtype='float32')
			y_V2_k      = np.zeros(self.dpc_imgs.shape[:3]+ (2, ), dtype='float32')

		t_start     = time.time()
		for iteration in range(tv_max_iter):
			# solve Least-Squares
			if use_gpu:
				AHy_k   = [AHy[0]+self.rho*(af.fft3(V2_k[:,:,:,0]-y_V2_k[:,:,:,0])+ af.conjg(fDx)*af.fft3(DV_k[:,:,:,0]+y_DV_k[:,:,:,0])\
														   				          + af.conjg(fDy)*af.fft3(DV_k[:,:,:,1]+y_DV_k[:,:,:,1])\
														   				          + af.conjg(fDz)*af.fft3(DV_k[:,:,:,2]+y_DV_k[:,:,:,2])),\
						   AHy[1]+self.rho*(af.fft3(V2_k[:,:,:,1]-y_V2_k[:,:,:,1])+ af.conjg(fDx)*af.fft3(DV_k[:,:,:,3]+y_DV_k[:,:,:,3])\
														   				          + af.conjg(fDy)*af.fft3(DV_k[:,:,:,4]+y_DV_k[:,:,:,4])\
														   				          + af.conjg(fDz)*af.fft3(DV_k[:,:,:,5]+y_DV_k[:,:,:,5]))]
			else:
				AHy_k   = [AHy[0]+self.rho*(F_3D(V2_k[:,:,:,0]-y_V2_k[:,:,:,0])+ fDx.conj()*F_3D(DV_k[:,:,:,0]+y_DV_k[:,:,:,0])\
														   				       + fDy.conj()*F_3D(DV_k[:,:,:,1]+y_DV_k[:,:,:,1])\
														   				       + fDz.conj()*F_3D(DV_k[:,:,:,2]+y_DV_k[:,:,:,2])),\
						   AHy[1]+self.rho*(F_3D(V2_k[:,:,:,1]-y_V2_k[:,:,:,1])+ fDx.conj()*F_3D(DV_k[:,:,:,3]+y_DV_k[:,:,:,3])\
														   				       + fDy.conj()*F_3D(DV_k[:,:,:,4]+y_DV_k[:,:,:,4])\
														   				       + fDz.conj()*F_3D(DV_k[:,:,:,5]+y_DV_k[:,:,:,5]))]
			V1_k[:,:,:,1],\
			V1_k[:,:,:,0] 	 = self._deconvTikhonov(AHA, AHy_k, determinant, use_gpu)

			# solve LASSO proximal step
			DV_k, DV_k_diff  = self._prox_LASSO(V1_k, y_DV_k, use_gpu)

			# solve Euclidean proximal step
			V2_k             = self._prox_projection(V1_k, V2_k, y_V2_k, boundary_constraint)

			# dual update
			y_DV_k          += DV_k_diff;
			y_V2_k          += V1_k - V2_k;

			print("elapsed time: {:5.2f} seconds, iteration : {:02d}/{:02d}".format(time.time()-t_start, iteration+1, tv_max_iter), end="\r")

		return V1_k[:,:,:,1], V1_k[:,:,:,0]

	def solve(self, method="Tikhonov", tv_max_iter=20, boundary_constraint={"real":"negative", "imag":"negative"}, use_gpu=False):
		'''
		_prox_LASSO performs the proximal operator and solves the LASSO problem with L1 norm for total variation regularization.
		Inputs:
			method   		    : select "Tikhonov" or "TV" deconvolution methods.
			tv_max_iter		    : If "TV" method is used, specify the number of iterations of the ADMM algorithm
			boundary_constraint : indicate whether to use positive or negative constraint on the scattering potential
			use_gpu   		    : flag to specify gpu usage
		Output:
			RI_obj			    : reconstructed 3D refractive index
		'''
		if use_gpu:
			globals()["af"] = __import__("arrayfire")

		AHA         = [(self.H_imag.conj()*self.H_imag).sum(axis=0), (self.H_imag.conj()*self.H_real).sum(axis=0),\
		               (self.H_real.conj()*self.H_imag).sum(axis=0), (self.H_real.conj()*self.H_real).sum(axis=0)]
		fIntensity  = F_3D(self.dpc_imgs).transpose(3, 0, 1, 2).astype('complex64')

		if method == "Tikhonov":
			print("="*10+" Solving 3D DPC with Tikhonov regularization "+"="*10)
			AHA[0]        += self.reg_imag
			AHA[3]        += self.reg_real
			AHy            = [(self.H_imag.conj()*fIntensity).sum(axis=0), (self.H_real.conj()*fIntensity).sum(axis=0)]
			if use_gpu:
				AHA = [af.to_array(AHA_i) for AHA_i in AHA]
				AHy = [af.to_array(AHy_i) for AHy_i in AHy]

			determinant    = AHA[0]*AHA[3]-AHA[1]*AHA[2]
			V_real, V_imag = self._deconvTikhonov(AHA, AHy, determinant, use_gpu)

		elif method == "TV":
			print("="*10+" Solving 3D DPC with total variation regularization and boundary value constraint "+"="*10)
			fDx            = np.zeros(self.dpc_imgs.shape[:3], dtype='complex64')
			fDy            = np.zeros(self.dpc_imgs.shape[:3], dtype='complex64')
			fDz            = np.zeros(self.dpc_imgs.shape[:3], dtype='complex64')
			fDx[0, 0, 0]   = 1.0; fDx[0, -1, 0] = -1.0; fDx = F_3D(fDx).astype('complex64')
			fDy[0, 0, 0]   = 1.0; fDy[-1, 0, 0] = -1.0; fDy = F_3D(fDy).astype('complex64')
			fDz[0, 0, 0]   = 1.0; fDz[0, 0, -1] = -1.0; fDz = F_3D(fDz).astype('complex64')
			AHA[0]        += self.rho*(fDx*fDx.conj() + fDy*fDy.conj() + fDz*fDz.conj() + 1.0)
			AHA[3]        += self.rho*(fDx*fDx.conj() + fDy*fDy.conj() + fDz*fDz.conj() + 1.0)
			if use_gpu:
				AHA = [af.to_array(AHA_i) for AHA_i in AHA]
				fDx = af.to_array(fDx)
				fDy = af.to_array(fDy)
				fDz = af.to_array(fDz)

			determinant    = AHA[0]*AHA[3]-AHA[1]*AHA[2]
			V_real, V_imag = self._deconvTV(AHA, determinant, fIntensity, fDx, fDy, fDz, tv_max_iter, boundary_constraint, use_gpu)
		RI_obj         = self._V2RI(V_real, V_imag)

		return RI_obj
