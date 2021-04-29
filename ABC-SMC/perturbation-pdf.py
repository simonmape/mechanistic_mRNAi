from numpy import pi, sqrt, exp, diag, array
from scipy.stats import multivariate_normal
from sys import argv, stdin, stdout, stderr

# Process arguments
if len(argv) != 7:
    stderr.write("Usage: {} STDEV1 STDEV2\n"
            "Return probability densities of joint normal distributions centered \n"
            "on given parameters for the perturbed parameter \n"
            "and with standard deviations STDEV1 and STDEV2\n".format(argv[0]))

    exit(1)

m_stdev = float(argv[1])
p_stdev = float(argv[2])
gm_stdev = float(argv[3])
gp_stdev = float(argv[4])
gb_stdev = float(argv[5])
population_size = float(argv[6])

# Read generation t and perturbed parameter from stdin
t = int(stdin.readline())

m_pe, p_pe , gm_pe, gp_pe, gb_pe, dummy_pe = [ float(num) for num in stdin.readline().split() ]

# Define normal pdf
normal_pdf = lambda mu, stdev, x: 1.0 / sqrt(2 * pi * stdev**2) * \
        exp( - (x - mu)**2 / (2.0 * stdev**2))

# Read parameters from stdin
m_array = []
p_array = []
gm_array = []
gp_array = []
gb_array = []
dummy_array = []

param_counter = 0
for line in stdin:
	m, p, gm, gp, gb, dummy = [ float(num) for num in line.split() ]
	m_array.append(m)
	p_array.append(p)
	gm_array.append(gm)
	gp_array.append(gp)
	gb_array.append(gb)
	dummy_array.append(dummy)
	param_counter+=1
	if param_counter == population_size:
		break
cov_array = []
cov_arguments = 0		
for line in stdin:
	cov_arguments +=1
	cov = [ float(num) for num in line.split() ][0]
	cov_array.append(cov)

covariance = array(cov_array).reshape((5,5))+ diag([0.00005,0.00005,0.00005,0.00005,0.00005])

# Return normal_pdf for every parameter
for m, p, gm, gp, gb, dummy in zip(m_array, p_array,gm_array,gp_array,gb_array,dummy_array):
	mean = array([m,p,gm,gp,gb])
	pdf = multivariate_normal.pdf([m_pe, p_pe , gm_pe, gp_pe, gb_pe],mean=mean,cov=covariance)
	#df1 = normal_pdf(m, m_stdev, m_pe)
	#pdf2 = normal_pdf(p, p_stdev, p_pe)
	#pdf3 = normal_pdf(gm, gm_stdev, gm_pe)
	#pdf4 = normal_pdf(gp, gp_stdev, gp_pe)
	#pdf5 = normal_pdf(gb, gb_stdev, gb_pe)
	#pdf = pdf1 * pdf2 * pdf3 * pdf4 * pdf5
	if pdf < 1e-308:
		pdf=0.0
	stdout.write("{:0.2f}\n".format(pdf))
