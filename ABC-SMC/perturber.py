from numpy.random import multivariate_normal
from sys import argv, stdin, stdout, stderr
from numpy import array, diag

# Process arguments
if len(argv) != 6:
    stderr.write("Usage: {} STDEV1 STDEV2\n"
            "Perturb two parameters by drawing "
            "from normal distributions\n"
            "centered on the given parameters "
            "and with standard deviations\n"
            "STDEV1 and STDEV2, respectively.\n".format(argv[0]))

    exit(1)

m_stdev = float(argv[1])
p_stdev = float(argv[2])
gm_stdev = float(argv[3])
gp_stdev = float(argv[4])
gb_stdev = float(argv[5])

# Read generation t and parameter from stdin
t = int(stdin.readline())

m, p , gm, gp, gb, dummy = [ float(num) for num in stdin.readline().split() ]

cov_array = []
cov_arguments = 0		
for line in stdin:
	cov_arguments +=1
	cov = [ float(num) for num in line.split() ][0]
	cov_array.append(cov)

covariance = array(cov_array).reshape((5,5))+diag([0.00005,0.00005,0.00005,0.00005,0.00005])

# Perturb parameter
mean = array([m, p , gm, gp, gb])
sample = multivariate_normal(mean,covariance,1)[0]
m_perturbed = sample[0]
p_perturbed = sample[1]
gm_perturbed = sample[2]
gp_perturbed = sample[3]
gb_perturbed = sample[4]

# Print perturbed parameter
stdout.write("{} {} {} {} {} {} \n".format(m_perturbed, p_perturbed,gm_perturbed,gp_perturbed,gb_perturbed,dummy))
