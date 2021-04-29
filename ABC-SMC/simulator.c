/**
 * SIMULATE FROM THE IBM
 * Simon Martina-Perez (martinaperez@maths.ox.ac.uk)
 * Binny function simulation by Alexander P. Browning (ap.browning@qut.edu.au)
 * Date: January 2021
 */

#include <math.h>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define JC_VORONOI_IMPLEMENTATION
#include "jc_voronoi.h"

// Universal constants
#define pi          3.14159265358979323846
#define min(a,b)    (((a) < (b)) ? (a) : (b))
#define max(a,b)    (((a) < (b)) ? (b) : (a))


/* SETTINGS */

// Max agents
const int Nmax = 5000;

// Density profile settings
const int nBins = 64;
double BinWidth = 8;

// Pair correlation settings
double PC_dr        = 1.75;
const int PC_nBins  = 20;


/**
 * RANDOM NUMBER GENERATOR
 */
int rand2();
int rseed = 0;
void srand2(int x) {
    rseed = x;
}
#define RAND_MAX2 ((1U << 31) - 1)
inline int rand2() {
    return rseed = (rseed * 1103515245 + 12345) & RAND_MAX2;
}




/**
 * UNIFORM NUMBER GENERATOR
 */

double sampleU() { 
    double random = ((double)rand2()) / ((double)RAND_MAX2);
    return random;
}


/**
 * VM SAMPLE
 */
double VM(double x, double mu, double kappa) {

    double out = exp(kappa * cos(x - mu));
    return out;

}
double sampleVM(double mu, double kappa) {

    // Sample from non-normalised density with rejection sampling
    double fmax = VM(mu,mu,kappa);

    double x = sampleU()*2*pi - pi;
    double y = VM(x,mu,kappa);
    double u = sampleU()*fmax;

    while (u > y) {
        x = sampleU()*2*pi - pi;
        y = VM(x,mu,kappa);
        u = sampleU()*fmax;
    }

    return x;

}


/**
 * PERIODIC SQUARE DISTANCE BETWEEN AGENTS
 */
double distance2(double x1, double x2, double y1, double y2, double L, double H) {

    double dx = fabs(x1 - x2);
    double dy = fabs(y1 - y2);

    dx = min(dx,L - dx);
    dy = min(dy,H - dy);

    double d = pow(dx,2) + pow(dy,2);
    return d;

}


/**
 * 1D PERIODIC DISPLACEMENT (x1 -> x2)
 */
double disp(double x1, double x2, double L) {

    double s = 0;

    double dx = x2 - x1;
    double adx = fabs(dx);
    double dxL = L - adx;

    if (adx < dxL) {
        s = dx;
    } else {
        if (x1 < L / 2) {
            s = -dxL;
        } else {
            s = dxL;
        }
    }

    return s;

}


/**
 * GAUSSIAN KERNEL
 */
double kernel(double r2, double sigma2, double gamma) {

    double y = 0;

    if (r2 < 9 * sigma2) {

        y = gamma * exp( -r2 / (2 * sigma2) );

    }

    return y;

}


/*
 * RANDOMLY CHOOSE
 */
int choose_agent(double* M, double M_tot) {

    int i = 0;
    double Mc = max(0,M[i]);
    double alpha = sampleU() * M_tot;
    while (alpha > Mc) {
        i += 1;
        Mc += max(0,M[i]);
    }
    return i;

}


/*
 * PERIODIC MODULUS
 */
double mod(double x, double L) {

    if (x <= 0) {
        x += L;
    } else if (x >= L) {
        x -= L;
    }

    return x;

}


/*
 * CALCULATE PAIR CORRELATION
 *  AVERAGE OF PERODIC PC BETWEEN Y < 400 AND Y > 1500
 */
int PairCorrelation(int N, double * X, double * Y, double L, double * PC) {
    
    // Initialise
    double PC1[PC_nBins];
    double PC2[PC_nBins];
    for (int i = 0; i < PC_nBins; i++) {
        PC1[i] = 0;
        PC2[i] = 0;
    }
    
    double x0, y0, x1, y1, dx, dy, dist;

    int N1 = 0;      // N < 108
    int N2 = 0;      // N > 404
    
    
    // Loop through agents 1 (x0,y0)
    for (int i = 0; i < N; i++) {
        x0 = X[i];
        y0 = Y[i];
        
        // y0 < 108 (and only look at second agent with y1 < 108)
        if (y0 < 108) {
            N1 += 1;
            
            // Loop through agents 2 (x1,y1)
            for (int j = 0; j < N; j++) { if (i != j) {
                x1 = X[j];
                y1 = Y[j];
                
                if (y1 < 108) {
                    
                    dx = fabs(x1 - x0);
                    dx = min(dx,L - dx);
                    
                    dy = fabs(y1 - y0);
                    dy = min(dy,108 - dy);
                    
                    dist = sqrt(dx * dx + dy * dy);
                    
                    // Bin distance
                    for (int k = 0; k < PC_nBins; k++) {
                        if (dist >= k * PC_dr && dist < (k+1) * PC_dr) {
                            PC1[k] += 1;
                            break;
                        }
                    }
                    
                } // end (y1 < 108)
                
            }} // end agent loop 2
                        
        } // end agent loop 1 (y0 < 108)
        
               
        // y0 > 404 (and only look at second agent with y1 > 404)
        if (y0 > 404) {
            N2 += 1;
            
            // Loop through agents 2 (x1,y1)
            for (int j = 0; j < N; j++) { if (i != j) {
                double x1 = X[j];
                double y1 = Y[j];
                
                if (y1 > 404) {
                    
                    dx = fabs(x1 - x0);
                    dx = min(dx,L - dx);
                    
                    dy = fabs(y1 - y0);
                    dy = min(dy,108 - dy);
                    
                    dist = sqrt(dx * dx + dy * dy);
                    
                    // Bin distance
                    for (int k = 0; k < PC_nBins; k++) {
                        if (dist >= k * PC_dr && dist < (k+1) * PC_dr) {
                            PC2[k] += 1;
                            break;
                        }
                    }
                    
                } // end (y1 > 404)
                
            }} // end agent loop 2
                        
        } // end agent loop 1 (y0 > 404)
           
    }
    
    
    // Average and scale to get PC
    for (int i = 0; i < PC_nBins; i++) {

        PC1[i] /= N1 * N1 / (L * 108) * pi * (pow((i+1) * PC_dr,2) - pow(i * PC_dr,2));
        PC2[i] /= N2 * N2 / (L * 108) * pi * (pow((i+1) * PC_dr,2) - pow(i * PC_dr,2));
        
        PC[i] = 0.5 * (PC1[i] + PC2[i]);
        
    }
    
    
    return 1;
    
}


/*
 * CALCULATE DENSITY PROFILE
 */
int DensityProfile(int N, double * Y, double * D) {
    
    double BinStart;
    double yloc;
    
    // Start D
    for (int i = 0; i < nBins; i++) {
        D[i] = 0;
    }
    
    // Loop through agents
    for (int agent = 0; agent < N; agent++) {
        yloc = Y[agent];
        
        // Determine appropriate bin
        for (int i = 0; i < nBins; i++) {
            BinStart = i * BinWidth;
            if (yloc > BinStart && yloc <= (BinStart + BinWidth)) {
                D[i] += 1;
            }
        }
               
    }
    
    return 1;
    
}
    

/* Link list node */
struct Node { 
    int data; 
    struct Node* next; 
}; 
  
/* Given a reference (pointer to pointer) to the head 
  of a list and an int, push a new node on the front 
  of the list. */
void push(struct Node** head_ref, int new_data) 
{ 
    /* allocate node */
    struct Node* new_node = (struct Node*)malloc(sizeof(struct Node)); 
  
    /* put in the data  */
    new_node->data = new_data; 
  
    /* link the old list off the new node */
    new_node->next = (*head_ref); 
  
    /* move the head to point to the new node */
    (*head_ref) = new_node; 
} 
  
/* Counts the no. of occurrences of a node 
   (search_for) in a linked list (head)*/
int count(struct Node* head, int search_for) 
{ 
    struct Node* current = head; 
    int count = 0; 
    while (current != NULL) { 
        if (current->data == search_for) 
            count++; 
        current = current->next; 
    } 
    return count; 
} 
  
int DegreeDistribution(int N, double * X, double * Y, double * Deg){ 
    struct Node* head = NULL;
    jcv_rect bounding_box = { {0.0f, 0.0f }, { 180.0f, 512.0f } };
    jcv_diagram diagram;
    jcv_point points[N];
    const jcv_site* sites;
    jcv_graphedge* graph_edge;

    memset(&diagram, 0, sizeof(jcv_diagram));
    for (int i=0; i<N; i++) {
        points[i].x = (float)X[i];
        points[i].y = (float)Y[i];
      }

    jcv_diagram_generate(N, (const jcv_point *)points, &bounding_box, 0, &diagram);

    sites = jcv_diagram_get_sites(&diagram);
    for (int i=0; i<diagram.numsites; i++) {
        int neighborCount =0;
        graph_edge = sites[i].edges;
        while (graph_edge) {
            neighborCount++;
            graph_edge = graph_edge->next;
        }
        push(&head,neighborCount);
      }
    
    for(int i=0; i<10;i++)
    {
        Deg[i] = (double)count(head,i)/diagram.numsites;
    }
    return 1;
}



double DistanceFunction(int N_obs, int N_sim, double *Den_obs, double *PC_obs, double *Den_sim, double  *PC_sim)
{
   
   /*
   Compute cell number part of error
   */
   double N_error = (double)pow(N_obs-N_sim,2)/pow(N_obs,2);
   
   /*
   Compute density part of error
   */
    double density_error =0;
    double density_denom = 0;
    for(int i =0; i<nBins;i++)
    {
        density_error += pow(Den_obs[i]-Den_sim[i],2);
    }
    
    for(int i=0; i<nBins;i++)
    {
        density_denom += pow(Den_obs[i],2);
    }
    
    /*
    Compute PC part of error
    */
    double PC_error = 0;
    double PC_denom =0;
    
    for(int i=0; i<PC_nBins;i++)
    {
        PC_error =+ pow(PC_obs[i]-PC_sim[i],2);
    }
    
    for(int i=0; i<PC_nBins;i++)
    {
        PC_denom += pow(PC_obs[i],2);
    }
    
    /*
    Compute degree distribution part of error
    */
    
    /*
    double Degree_error = 0;
    
    for(int i=0; i<10;i++)
    {
        if(Deg_obs[i]!=0 && Deg_sim[i]!=0)
        {
            Degree_error =+ Deg_obs[i]*log(Deg_obs[i]/Deg_sim[i]);
        }
        
    }
    */
    return N_error + density_error/density_denom + PC_error/PC_denom;
}
    





/**
 * SIMULATE BINNY MODEL
 */
int Binny(double m, double p, double gm, double gp, double gb, double s2, double mu_s, double L, double H, int N0, double T, double * X, double * Y)
{
    
    // INITIALISE
    int     N       = N0;
    double  t       = 0;
    double M[Nmax];
    double P[Nmax];
    double Bx[Nmax];
    double By[Nmax];
    
    // Loop through each agent
    for (int i = 0; i < N0; i++) {
        
        double x1 = X[i];
        double y1 = Y[i];
        
        double MrS = m;
        double PrS = p;
        double BxS = 0;
        double ByS = 0;
        
        // Loop through other agents
        for (int j = 0; j < N; j++) { if (i != j) {
            
            double x2 = X[j];
            double y2 = Y[j];
            
            double r2 = distance2(x1,x2,y1,y2,L,H);
            
            double b_ = kernel(r2,s2,gb) / s2;
            MrS      -= kernel(r2,s2,gm);
            PrS      -= kernel(r2,s2,gp);
            
            if (b_ != 0) {
                // Bx is b_ * (disp from them to us)
                BxS += b_ * disp(x2,x1,L);
                ByS += b_ * disp(y2,y1,H);
            }
            
        }}
        
        M[i] = MrS;
        P[i] = PrS;
        Bx[i] = BxS;
        By[i] = ByS;
        
    }

    // END INITIALISE

    
    // LOOP THROUGH TIME
    while (t < T && N < Nmax) {
        
        // CALCULATE TOTAL EVENT RATES
        double M_tot = 0;
        double P_tot = 0;
        for (int i = 0; i < N; i++) {
            M_tot += max(0,M[i]);
            P_tot += max(0,P[i]);
        }
       
        // SAMPLE TIMESTEP
        double tau = -log(sampleU()) / (M_tot + P_tot);
        t += tau;
        
        // STOP IF NEXT EVENT OCCURS AFTER t = T
        if (t > T) {
            break;
        }
        
        // DECIDE EVENT
        double alpha = sampleU() * (M_tot + P_tot);
        
        // MOVEMENT
        if (alpha < M_tot) {
            // CHOOSE AGENT
            int i = choose_agent(M,M_tot);
            
            // LOCATION
            double xc = X[i];
            double yc = Y[i];
            
            // INCREASE RATES OF SURROUNDING AGENTS
            for (int j = 0; j < N; j++) { if (i != j) {
                
                double x2 = X[j];
                double y2 = Y[j];
                
                double r2 = distance2(xc,x2,yc,y2,L,H);
                double m_ = kernel(r2,s2,gm);
                double p_ = kernel(r2,s2,gp);
                double b_ = kernel(r2,s2,gb) / s2;
                
                if (m_ != 0) {
                    M[j] += m_;
                }
                if (p_ != 0) {
                    P[j] += p_;
                }
                if (b_ != 0) {
                    // Bx is b_ * (disp from them to us)
                    Bx[j] -= b_ * disp(xc,x2,L);
                    By[j] -= b_ * disp(yc,y2,H);
                }
                
            }}
            // END INCREASE RATES
            
            
            // MOVE SOMEWHERE, INCLUDE BIAS
            double md       = mu_s;
            double Bx_i     = Bx[i];
            double By_i     = By[i];
            double vm_mu    = atan2(By_i,Bx_i);
            double vm_kappa = sqrt(pow(Bx_i,2) + pow(By_i,2));
            
            double theta    = sampleVM(vm_mu,vm_kappa);
            
            double xp       = mod(xc + md * cos(theta),L);
            double yp       = mod(yc + md * sin(theta),H);
            
            // UPDATE RATES OF AGENT
            double MrS      = m;
            double PrS      = p;
            double BxS      = 0;
            double ByS      = 0;
            
            for (int j = 0; j < N; j++) { if(i != j) {

                double x2 = X[j];
                double y2 = Y[j];

                double r2 = distance2(xp,x2,yp,y2,L,H);
                double m_ = kernel(r2,s2,gm);
                double p_ = kernel(r2,s2,gp);
                double b_ = kernel(r2,s2,gb) / s2;

                if (m_ != 0) {
                    M[j] -= m_;
                    MrS  -= m_;
                }
                if (p_ != 0) {
                    P[j] -= p_;
                    PrS  -= p_;
                }
                if (b_ != 0) {
                    // Bx is b_ * (disp from them to us)
                    double sx = disp(x2,xp,L);
                    double sy = disp(y2,yp,H);
                    
                    // This agents bias
                    BxS      += b_ * sx;
                    ByS      += b_ * sy;
                    
                    // Other agents bias (displacement -ve)
                    Bx[j]    -= b_ * sx;
                    By[j]    -= b_ * sy;
                }
               
            }}
            // END UPDATE RATES
            
            
            // "MOVE" AGENT
            X[i]    = xp;
            Y[i]    = yp;
            M[i]    = MrS;
            P[i]    = PrS;
            Bx[i]   = BxS;
            By[i]   = ByS;
            
            
       // PROLIFERATION
       } else {
            
            // CHOOSE AGENT
            int i = choose_agent(P,P_tot);
            
            // LOCATION
            double xc = X[i];
            double yc = Y[i];
            
            // NEW LOCATION (USE BIAS)
            double md       = mu_s;
            double Bx_i     = Bx[i];
            double By_i     = By[i];
            double vm_mu    = atan2(By_i,Bx_i);
            double vm_kappa = sqrt(pow(Bx_i,2) + pow(By_i,2));
            
            double theta    = sampleVM(vm_mu,vm_kappa);
            
            double xp       = mod(xc + md * cos(theta),L);
            double yp       = mod(yc + md * sin(theta),H);
            
            // UPDATE RATES
            double MrS = m;
            double PrS = p;
            double BxS = 0;
            double ByS = 0;
            
            for (int j = 0; j < N; j++) {
             
                double x2 = X[j];
                double y2 = Y[j];

                double r2 = distance2(xp,x2,yp,y2,L,H);
                double m_ = kernel(r2,s2,gm);
                double p_ = kernel(r2,s2,gp);
                double b_ = kernel(r2,s2,gb) / s2;

                if (m_ != 0) {
                    M[j] -= m_;
                    MrS  -= m_;
                }
                if (p_ != 0) {
                    P[j] -= p_;
                    PrS  -= p_;
                }
                if (b_ != 0) {

                    // Bx is b_ * (disp from them to us)
                    double sx = disp(x2,xp,L);
                    double sy = disp(y2,yp,H);
                    
                    // This agents bias
                    BxS      += b_ * sx;
                    ByS      += b_ * sy;
                    
                    // Other agents bias (displacement -ve)
                    Bx[j]    -= b_ * sx;
                    By[j]    -= b_ * sy;
                    
                }
                
            }
            // END UPDATE RATES
            
            // "CREATE" NEW AGENT
             X[N]    = xp;
             Y[N]    = yp;
             M[N]    = MrS;
             P[N]    = PrS;
             Bx[N]   = BxS;
             By[N]   = ByS;
             N += 1;
            
        } // END PROLIFERATION

    }
    return N;
}
// END BINNY FUNCTION


int main(int argc, char *argv[])
{
    //Process arguments
    double s2_ = atof(argv[1]);
    double mu_s_= atof(argv[2]);
    double L_ = atof(argv[3]);
    double H_ = atof(argv[4]);
    int batchSize = atoi(argv[5]);
    
    double m_;
    double p_;
    double gm_;
    double gp_;
    double gb_;
    int seed;
    scanf("%lf %lf %lf %lf %lf %d",&m_,&p_,&gm_,&gp_,&gb_,&seed);
    //Random seed
    
    srand2(seed);
    
    
    /*
     Process text file with file names of initial and final
     conditions corresponding to each of the experiments
     */
    char placeholder[2000][50];
    char initialConditions[1000][50];
    char endConditions[1000][50];
    FILE *initialFileNames = fopen(argv[6],"r");
    int k =0;
    while (fscanf(initialFileNames, "%s", placeholder[k]) != EOF)
    {
        ++k;
    }
    fclose(initialFileNames);
    
    
    for(int i=0;i<k/2;i++) 
    {
        strcpy(initialConditions[i],placeholder[i]);
        strcpy(endConditions[i],placeholder[k/2+i]);
    }
    
    int number_of_samples = k/2;   
    double epsilon =0;
    for(int batch =0; batch < batchSize;batch++)
    {
        /*
        Select random batch for computing the distance
        between simulated and observed
        */
        int datapoint = (int) floor(sampleU()*number_of_samples);

        /*
        Read initial condition from txt file and
        initialize arrays
         */
        double X[Nmax];
        double Y[Nmax];
        double nums[10000] = {0};
        k = 0;
        FILE *datafile = fopen(initialConditions[datapoint],"r");

        while (fscanf(datafile, "%lf", &nums[k]) != EOF)
        {
            ++k;
        }
        fclose(datafile);
        
        /*
        Assuminng that the data is formatted such that all the
        X are provided first, and then the Y are given in the
        txt file
         */
        for(int i=0;i<k/2;i++)
        {
            X[i] = nums[i] -470;
            Y[i] = nums[i+k/2];
        }
        
        
        int N0_ = k/2; //Total number of cells in the beginning
        int N_sim = Binny(m_,p_,gm_,gp_,gb_,s2_,mu_s_,L_,H_,N0_,24,X,Y);
        
        /*
        Load the observed data for this batch
         */
        double X_obs[Nmax];
        double Y_obs[Nmax];
        double Density_obs[nBins];
        double PC_obs[PC_nBins];
        double Degrees_obs[10];        
        double nums2[10000] = {0};
        
        FILE *obsfile = fopen(endConditions[datapoint],"r");
        
        k=0;
        while (fscanf(obsfile, "%lf", &nums2[k]) != EOF)
        {
            ++k;
        }
        fclose(obsfile);
        
        int N_obs = (k-(nBins+PC_nBins+10))/2; //the last 94 entries 
                                            //correspond to density/PC/degrees
                                           
        /*
        Again assuming X and Y are ordered in txt file
         */
        for(int i=0;i<N_obs;i++)
        {
            X_obs[i] = nums2[i];
            Y_obs[i] = nums2[i+N_obs];
        }
        
        /*
        Load in the density data 
        */
        for(int i=0;i<nBins;i++)
        {
            Density_obs[i] = nums2[i+2*N_obs];
        }
        
        
        /*
        Load in the PC data
        */
        for(int i=0;i<PC_nBins;i++)
        {
            PC_obs[i] = nums2[i+2*N_obs+nBins];
        }
        
        /*
        Load in the degree distribution data
        */
        //for(int i=0;i<10;i++)
        //{
        //    Degrees_obs[i] = nums2[i+2*N_obs+nBins+PC_nBins];
        //}
        
        /*
        Create variables for simulated summary statistics
        */
        double Density_sim[nBins];
        double PC_sim[PC_nBins];
        double Degrees_sim[10];
                
        /*
        Compute simulated density profile
        */
        DensityProfile(N_sim, Y, Density_sim);
        
        /*
        Compute simulated PC
        */
        int Length = 180; //Hardcoded the length in pixels here.
        PairCorrelation(N_sim, X, Y, Length, PC_sim);
        
        /*
        Compute simulated degree distribution
        */
        //DegreeDistribution(N_sim, X, Y,Degrees_sim);
        
        epsilon += DistanceFunction(N_obs, N_sim, Density_obs, PC_obs, Density_sim, PC_sim);
             
    }
    if(isnan(epsilon))
    	epsilon = 5000;
    printf("%lf \n", epsilon/batchSize);
    return 0;
}
