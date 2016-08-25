// Written by Jianxin Wu (jxwu @ ntu.edu.sg / wujx2001 @ gmail.com)
// 2011
//
// Some routines from LIBLINEAR are revised to be used in this software
// A copy of the LIBLINEAR licence is provided in this package.
// 
// This software is provided for research or academic purpose only.
// No warrenty, explicit or implicit, is given or implied.

#include <cstdio>
#include <cstring>
#include <cmath>
#include <cerrno>
#include <cassert>

#include <iostream>
#include <algorithm>
#include <numeric>

#include <sys/timeb.h>

//#define __USE_DOUBLE

#ifdef __USE_DOUBLE
	typedef double NUMBER;
	const double BIGVALUE = HUGE_VAL;
#else
	typedef float NUMBER;
	const float BIGVALUE = HUGE_VALF;
#endif

// linear regression interpolation points
const NUMBER Xvalue[3] = {0.01, 0.06, 0.75}; 
NUMBER Xvalue2[3];

const NUMBER LogOffset = 0.05; // x --> log(x+LogOffset) in regression
// Regression matrix; X^{-1} in the algorithm
const NUMBER RegMatrix[3][3] =     {
	{ 0.3137,   -0.5220,    1.2083 },
    { 1.5480,   -2.5249,    0.9769 },
    { 0.6369,   -0.8315,    0.1946 }
};

bool verbose = true;

// Compute classification accuracy
// 'acc': overall accuracy
// 'avgacc': average accuracy -- refer to the manual
void ComputeAccuracies(const int size,const int nr_class,const int* gt,const int* pd,NUMBER& acc,NUMBER& avgacc)
{
	int* confusion = new int[(unsigned long)nr_class*nr_class];
	std::fill(confusion,confusion+nr_class*nr_class,0.0);
	acc = 0;
	for(int i=0;i<size;i++) confusion[gt[i]*nr_class+pd[i]]++;
	for(int i=0;i<nr_class;i++) acc += confusion[i*nr_class+i];
	acc = acc / size * 100;
	avgacc = 0;
	for(int i=0;i<nr_class;i++)
	{
		int sum = std::accumulate(confusion+i*nr_class,confusion+(i+1)*nr_class,0);
		NUMBER oneacc = confusion[i*nr_class+i]*1.0/sum;
		if(sum==0) oneacc = 1;
		if(verbose) std::cout<<confusion[i*nr_class+i]<<'/'<<sum<<'='<<100.0*confusion[i*nr_class+i]/sum<<'%'<<std::endl;
		avgacc += oneacc;
	}
	avgacc = avgacc / nr_class * 100;
	delete[] confusion;
}

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
static char *line = NULL;
static int max_line_len;

static char* readline(FILE *input)
{
	if(fgets(line,max_line_len,input) == NULL) return NULL;

	int len;
	while(strrchr(line,'\n') == NULL)
	{
		max_line_len *= 2;
		line = (char *) realloc(line,max_line_len);
		len = (int) strlen(line);
		if(fgets(line+len,max_line_len-len,input) == NULL) break;
	}
	return line;
}

void exit_input_error(int line_num)
{
	std::cerr<<"Wrong input format at line "<<line_num<<std::endl;
	exit(1);
}

class model
{
public:
	NUMBER* _dec_values;
public:
	int nr_class;
	int nr_feature;
	NUMBER p; // p should be negative
	NUMBER *w;
	int *label;		/* label of each class */
public:
	model();
	~model();
public:
	int PredictValues(const int* ix,const NUMBER* vx,NUMBER *dec_values);
	int Predict(const int* ix,const NUMBER* vx);
	int FindLabel(const int newl) const;
};

model::model()
	:_dec_values(NULL),nr_class(0),nr_feature(0),p(1.0),w(NULL),label(NULL)
{
}

model::~model()
{
	if(_dec_values) { delete[] _dec_values; _dec_values = NULL; }
	nr_class = nr_feature = 0;
	p = 1.0;
	delete[] w; w = NULL;
	delete[] label; label = NULL;
}

int model::FindLabel(const int newl) const
{
	assert(nr_class>0); // aka, GroupClasses() has been called
	for(int i=0;i<nr_class;i++) if(label[i]==newl) return i;
	return -1;
}

class problem
{
private:
	int num_class;
	int* labels;
private:
	size_t allocated;
	int* index_buf;
	NUMBER* value_buf;
private:
	int l, n;
	int *y;
	int** indexes;
	NUMBER** values;
public:
	problem();
	~problem();
	void Clear();
public:
	void Load(const char* filename,const NUMBER bias=0,const int maxdim=0);
	void Train(model& model_,const NUMBER C,const NUMBER p);
	void Clone(problem& dest);
	NUMBER CrossValidation(const int nr_fold,const NUMBER C,const NUMBER p);
public:
	int FindLabel(const int newl) const;
	int GetNumClass() { return num_class; }
	int GetNumExamples() { return l; }
	int GetLabel(const int index) { assert(index>=0 && index<l); return y[index]; }
	const int* GetFeatureIndexes(const int index) { assert(index>=0 && index<l); return indexes[index]; };
	const NUMBER* GetFeatureValues(const int index) { assert(index>=0 && index<l); return values[index]; };
private:
	void GroupClasses(int** start_ret=NULL, int** count_ret=NULL,int* perm=NULL);
	void Solve_l2r_l1l2_svc(NUMBER* w,const NUMBER C,const NUMBER p,const int solver_type,const NUMBER* fvalues,const NUMBER* logxplus) const;
};

problem::problem()
	:num_class(0),labels(NULL),allocated(0),index_buf(NULL),value_buf(NULL),l(0),n(0),y(NULL),indexes(NULL),values(NULL)
{
}

problem::~problem()
{
	Clear();
}

void problem::Clear()
{
	allocated = 0;
	l = n = num_class = 0;
	delete[] labels; labels = NULL;
	if(index_buf) { free(index_buf); index_buf = NULL; }
	if(value_buf) { free(value_buf); value_buf = NULL; }
	delete[] y; y = NULL;
	delete[] indexes; indexes = NULL;
	delete[] values; values = NULL;
}

void problem::Clone(problem& dest)
{
	dest.Clear();
	dest.allocated = allocated;
	dest.l = l; dest.n = n; dest.num_class = num_class;
	dest.labels = new int[num_class];
	std::copy(labels,labels+num_class,dest.labels);
	dest.index_buf = (int*)malloc(sizeof(int)*(unsigned long)allocated);
	std::copy(index_buf,index_buf+allocated,dest.index_buf);
	dest.value_buf = (NUMBER*)malloc(sizeof(NUMBER)*(unsigned long)allocated);
	std::copy(value_buf,value_buf+allocated,dest.value_buf);
	dest.y = new int[l];
	std::copy(y,y+l,dest.y);
	dest.indexes = new int*[l];
	for(int i=0;i<l;i++) dest.indexes[i] = dest.index_buf + (indexes[i]-index_buf);
	dest.values = new NUMBER*[l];
	for(int i=0;i<l;i++) dest.values[i] = dest.value_buf + (values[i]-value_buf);
}

void problem::Load(const char* filename,const NUMBER bias,const int maxdim)
{	// revised from the LIBLINEAR/LIBSVM read_problem() function
	timeb TimingMilliSeconds;
	ftime(&TimingMilliSeconds);
	Clear();
	
	FILE *fp = fopen(filename,"r");
	if(fp == NULL)
	{
		fprintf(stderr,"can't open input file %s\n",filename);
		exit(1);
	}

	max_line_len = 102400;
	line = Malloc(char,max_line_len);
	l = 0;
	allocated = 0;
	while(readline(fp)!=NULL)
	{
		char *p = strtok(line," \t"); // label
		// features
		while(1)
		{
			p = strtok(NULL," \t");
			if(p == NULL || *p == '\n') // check '\n' as ' ' may be after the last feature
				break;
			allocated++;
		}
		allocated++;
		l++;
	}
	rewind(fp);

	if(bias>0) allocated += l;
	index_buf = new int[allocated];
	value_buf = new NUMBER[allocated];
	y = new int[l];
	indexes = new int*[l];
	values = new NUMBER*[l];
	
	int max_index = 0;
	long int j=0;
	char *endptr;
	char *idx, *val, *label;
	for(int i=0;i<l;i++)
	{
		int inst_max_index = 0; // strtol gives 0 if wrong format
		readline(fp);
		indexes[i] = &index_buf[j];
		values[i] = &value_buf[j];
		label = strtok(line," \t");
		y[i] = (int) strtol(label,&endptr,10);
		if(endptr == label) exit_input_error(i+1);
		
		while(1)
		{
			if(bias>0)
			{
				index_buf[j] = 1;
				value_buf[j] = bias;
			}
			
			idx = strtok(NULL,":");
			val = strtok(NULL," \t");

			if(val == NULL)
				break;

			errno = 0;
			index_buf[j] = (int) strtol(idx,&endptr,10)+(bias>0);
			if(endptr == idx || errno != 0 || *endptr != '\0' || index_buf[j] <= inst_max_index)
				exit_input_error(i+1);
			else
				inst_max_index = index_buf[j];

			errno = 0;
			value_buf[j] = (NUMBER)strtod(val,&endptr);
			if(value_buf[j]<0)
				value_buf[j] = 0.001;
			//else if(value_buf[j]>=1)
				//value_buf[j] = 0.999;

			if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
				exit_input_error(i+1);

			++j;
		}
		if(inst_max_index > max_index) max_index = inst_max_index;
		index_buf[j++] = -1;
	}
	n=max_index;
	if(maxdim>0 && n>maxdim) n=maxdim;

	fclose(fp);
	free(line);
	line = NULL;
	
	struct timeb now;
    ftime(&now);
    std::cout<<"Dataset loaded in "<<int( (now.time-TimingMilliSeconds.time)*1000+(now.millitm-TimingMilliSeconds.millitm) )<<" msec."<<std::endl;
}

void problem::GroupClasses(int** start_ret,int** count_ret,int* perm)
{   // modified from group_classes() function in LIBLINEAR by Jianxin Wu
	int max_nr_class = 16;
	int nr_class = 0;
	int *label = (int*)malloc((unsigned long)max_nr_class*sizeof(int));
	int *count = (int*)malloc((unsigned long)max_nr_class*sizeof(int));
	int *data_label = new int[l];
	int i;

	for(i=0;i<l;i++)
	{
		int this_label = y[i];
		int j;
		for(j=0;j<nr_class;j++) if(this_label == label[j]) { ++count[j]; break; }
		data_label[i] = j;
		if(j == nr_class)
		{
			if(nr_class == max_nr_class)
			{
				max_nr_class *= 2;
				label = (int *)realloc(label,max_nr_class*sizeof(int));
				count = (int *)realloc(count,max_nr_class*sizeof(int));
			}
			label[nr_class] = this_label;
			count[nr_class] = 1;
			++nr_class;
		}
	}

	int *start = new int[nr_class];
	bool delete_here = false;
    if(perm==NULL)
    {
		assert(start_ret==NULL && count_ret==NULL);
		perm = new int[l];
		delete_here = true;
	}
	start[0] = 0;
	for(i=1;i<nr_class;i++) start[i] = start[i-1]+count[i-1];
	for(i=0;i<l;i++)
	{
		perm[start[data_label[i]]] = i;
		++start[data_label[i]];
	}
	start[0] = 0;
	for(i=1;i<nr_class;i++) start[i] = start[i-1]+count[i-1];
	
	num_class = nr_class;
	if(labels) delete[] labels;
	labels = new int[num_class];
	std::copy(label,label+num_class,labels);
    free(label);
    
    if(delete_here)
    {
		free(count);
		delete[] start;
		delete[] perm;
	}
	else
	{
		*start_ret = start;
		*count_ret = count;
	}
	delete[] data_label;
}

// newl is labels used in the training / testing set, which can be 
//   discontinuous, e.g., 1, 5, -3, etc
// Now convert 'newl' to the labels used internally in Train() function
//   The internal labels should be 0, 1, 2, ..., num_class-1
int problem::FindLabel(const int newl) const
{
	assert(num_class>0); // aka, GroupClasses() has been called
	for(int i=0;i<num_class;i++) if(labels[i]==newl) return i;
	return -1;
}

void problem::Train(model& model_,const NUMBER C,const NUMBER p)
{
	timeb TimingMilliSeconds;
	ftime(&TimingMilliSeconds);
	
	int *start = NULL;
	int *count = NULL;
	int *perm = new int[l];
	GroupClasses(&start,&count,perm);
	
	for(int i=0;i<3;i++) Xvalue2[i] = 0.5*pow(Xvalue[i],p);
	const NUMBER invp = 1.0/p;
	NUMBER* fvalues = new NUMBER[(unsigned long)allocated*2]; assert(fvalues != NULL);
	// we assume for the small value xvalue[0]=0.01, any feature value will give the same kernel similarity 0.01
	// this approximation sometimes increases accuracy, and saves memory (fvalues use only 2/3 memory) and time
	if(p==-1)
	{
		for(size_t i=0;i<allocated;i++)
		{
			if(index_buf[i]==-1) continue;
			const NUMBER v = 0.5/value_buf[i];
			fvalues[2*i] = 1.0/(Xvalue2[1]+v);
			fvalues[2*i+1] = 1.0/(Xvalue2[2]+v);
		}
	}
	else
	{
		for(size_t i=0;i<allocated;i++)
		{
			if(index_buf[i]==-1) continue;
			const NUMBER v = 0.5*pow(value_buf[i],p);
			fvalues[2*i] = pow(Xvalue2[1]+v,invp);
			fvalues[2*i+1] = pow(Xvalue2[2]+v,invp);
		}
	}
	NUMBER* logxplus = new NUMBER[allocated]; assert(logxplus!=NULL);
	for(size_t i=0;i<allocated;i++) if(index_buf[i]!=-1) logxplus[i] = log(value_buf[i]+LogOffset);

	model_.nr_feature=n;
	model_.nr_class=num_class;
	if(model_.label) delete[] model_.label;
	model_.label = new int[num_class];
	for(int i=0;i<num_class;i++) model_.label[i] = labels[i];
	if(model_._dec_values) delete[] model_._dec_values;
	model_._dec_values = new NUMBER[model_.nr_class];	
	
	problem sub_prob;
	sub_prob.l = l;
	sub_prob.n = n;
	sub_prob.values = new NUMBER*[l];
	sub_prob.indexes = new int*[l];
	sub_prob.y = new int[l];
	sub_prob.index_buf = index_buf; // This is required to compute offset for 'logxplus'
	for(int k=0; k<sub_prob.l; k++)
	{
		sub_prob.values[k] = values[perm[k]];
		sub_prob.indexes[k] = indexes[perm[k]];
	}

	if(model_.w) delete[] model_.w;
	if(num_class == 2)
	{
		model_.w = new NUMBER[(unsigned long)n*3];
		int e0 = start[0]+count[0];
		int k=0;
		for(; k<e0; k++) sub_prob.y[k] = +1;
		for(; k<sub_prob.l; k++) sub_prob.y[k] = -1;
		sub_prob.Solve_l2r_l1l2_svc(model_.w,C,p,1,fvalues,logxplus);
	}
	else
	{
		model_.w = new NUMBER[(unsigned long)n*num_class*3];
		NUMBER* w = new NUMBER[(unsigned long)n*3];
		for(int i=0;i<num_class;i++)
		{
			int si = start[i];
			int ei = si+count[i];
			unsigned long k=0;
			for(; k<si; k++) sub_prob.y[k] = -1;
			for(; k<ei; k++) sub_prob.y[k] = +1;
			for(; k<sub_prob.l; k++) sub_prob.y[k] = -1;
			sub_prob.Solve_l2r_l1l2_svc(w,C,p,1,fvalues,logxplus);
			// the old way is to save the classifier for one class together
			// 1. num_class blocks, one block for one class
			// 2. n sub-blocks, one sub-block (3 values) for one feature
			// std::copy(w,w+n*3,model_.w+i*n*3); // the old way
			
			// Changed back to LIBLINEAR order for better cache consistency
			// 1. n blocks, one block for one feature
			// 2. num_class sub-blocks, one sub-block (3 values) for one class
			for(k=0;k<n;k++)
			{
				model_.w[k*3*num_class + i*3] = w[k*3];
				model_.w[k*3*num_class + i*3 + 1] = w[k*3+1];
				model_.w[k*3*num_class + i*3 + 2] = w[k*3+2];
			}
		}
		delete[] w;
	}
	sub_prob.index_buf = NULL; // This is necessary to avoid double free of 'index_buf'

	delete[] perm;
	delete[] start;
	free(count);
	delete[] fvalues;
	delete[] logxplus;
	
	struct timeb now;
    ftime(&now);
    std::cout<<"Finished in "<<int( (now.time-TimingMilliSeconds.time)*1000+(now.millitm-TimingMilliSeconds.millitm) )<<" msec."<<std::endl;
}

void problem::Solve_l2r_l1l2_svc(NUMBER* w,NUMBER C,const NUMBER p,const int solver_type,const NUMBER* fvalues,const NUMBER* logxplus) const
{
	NUMBER* woffset3 = w - 3;
	const NUMBER eps = 0.1;
	int i, s, iter = 0;
	NUMBER d, G;
	NUMBER *QD = new NUMBER[l];
	int max_iter = 1000;
	int *index = new int[l];
	NUMBER *alpha = new NUMBER[l];
	int *newy = new int[l];
	int active_size = l;

	// PG: projected gradient, for shrinking and stopping
	NUMBER PG;
	NUMBER PGmax_old = BIGVALUE;
	NUMBER PGmin_old = -BIGVALUE;
	NUMBER PGmax_new, PGmin_new;
	NUMBER Cp = C, Cn = C;
	
	NUMBER diag_p = 0.5/Cp, diag_n = 0.5/Cn;
	NUMBER upper_bound_p = BIGVALUE, upper_bound_n = BIGVALUE;
	if(solver_type == 2) // L2R_L1LOSS_SVC_DUAL
	{
		diag_p = 0; diag_n = 0;
		upper_bound_p = Cp; upper_bound_n = Cn;
	}

	std::fill(w,w+n*3,0.0);
	for(i=0; i<l; i++)
	{
		alpha[i] = 0;
		if(y[i] > 0)
		{
			newy[i] = +1; 
			QD[i] = diag_p;
		}
		else
		{
			newy[i] = -1;
			QD[i] = diag_n;
		}
		const int* ix = indexes[i];
		const NUMBER* vx = values[i];
		// if efficiency is really really important, this line can be moved
		// into the 'Train' method
		while(*ix++ != -1) QD[i] += *vx++;
		index[i] = i;
	}

	while(iter < max_iter)
	{
		PGmax_new = -BIGVALUE;
		PGmin_new = BIGVALUE;

		for(i=0;i<active_size;i++)
		{
			// use rand_r for thread safety
			int j = i+rand()%(active_size-i);
			std::swap(index[i], index[j]);
		}

		for(s=0;s<active_size;s++)
		{
			i = index[s];
			G = 0;
			int yi = newy[i];

			const int* ix = indexes[i];
			const NUMBER* logx = logxplus + (ix-index_buf);
			while(*ix != -1)
			{
				NUMBER* pw = woffset3 + *ix++ * 3;  // w + (*ix++ - 1)*3
				const NUMBER logxvalue = *logx++;
				G += (pw[2]*logxvalue + pw[1])*logxvalue + pw[0];
			}
			G = G*yi-1;

			if(yi == 1)
			{
				C = upper_bound_p; 
				G += alpha[i]*diag_p; 
			}
			else 
			{
				C = upper_bound_n;
				G += alpha[i]*diag_n; 
			}

			PG = 0;
			if(alpha[i] == 0)
			{
				if(G > PGmax_old)
				{
					active_size--;
					std::swap(index[s], index[active_size]);
					s--;
					continue;
				}
				else if(G < 0)
					PG = G;
			}
			else if(alpha[i] == C)
			{
				if(G < PGmin_old)
				{
					active_size--;
					std::swap(index[s], index[active_size]);
					s--;
					continue;
				}
				else if(G > 0)
					PG = G;
			}
			else
				PG = G;

			PGmax_new = std::max(PGmax_new, PG);
			PGmin_new = std::min(PGmin_new, PG);

			if(fabs(PG) > 1.0e-12)
			{
				NUMBER alpha_old = alpha[i];
				alpha[i] = std::min(std::max(alpha[i] - G/QD[i], (NUMBER)0.0), C);
				d = (alpha[i] - alpha_old)*yi;
				const int* ix = indexes[i];
				const NUMBER* pf = (NUMBER*)(fvalues + (ix-index_buf)*2);
				while(*ix != -1)
				{
					NUMBER* pw = woffset3 + *ix++ * 3; // w + (*ix++ - 1)*3
					const NUMBER f1 = *pf++; const NUMBER f2 = *pf++;
					*pw++ += d*(RegMatrix[0][0]*Xvalue[0] + RegMatrix[0][1]*f1 + RegMatrix[0][2]*f2);
					*pw++ += d*(RegMatrix[1][0]*Xvalue[0] + RegMatrix[1][1]*f1 + RegMatrix[1][2]*f2);
					*pw++ += d*(RegMatrix[2][0]*Xvalue[0] + RegMatrix[2][1]*f1 + RegMatrix[2][2]*f2);
				}
			}
		}

		iter++;
		if(iter % 10 == 0 && verbose) (std::cout<<'.').flush();

		if(PGmax_new - PGmin_new <= eps)
		{
			if(active_size == l)
				break;
			else
			{
				active_size = l;
				if(verbose) (std::cout<<"*.").flush();
				PGmax_old = BIGVALUE;
				PGmin_old = -BIGVALUE;
				continue;
			}
		}
		PGmax_old = PGmax_new;
		PGmin_old = PGmin_new;
		if (PGmax_old <= 0)
			PGmax_old = BIGVALUE;
		if (PGmin_old >= 0)
			PGmin_old = -BIGVALUE;
	}

	if(verbose) std::cout<<std::endl<<"optimization finished, #iter = "<<iter<<std::endl;
	int nSV = 0;
	for(i=0;i<l;i++) if(alpha[i]>0) ++nSV;
	if(verbose) std::cout<<"nSV = "<<nSV<<std::endl;

	delete[] QD;
	delete[] alpha;
	delete[] newy;
	delete[] index;
}

int model::PredictValues(const int* ix,const NUMBER* vx,NUMBER *dec_values)
{
	unsigned long n = nr_feature;
	unsigned long i;
	unsigned long nr_w;
	if(nr_class==2)
		nr_w = 1;
	else
		nr_w = nr_class;

	for(i=0;i<nr_w;i++) dec_values[i] = 0;
	int px = 0;
	for(; ix[px]!=-1; px++)
	{
		// the dimension of testing data may exceed that of training
		if(ix[px]<=n)
		{
			const NUMBER logxvalue = log(vx[px]+LogOffset);
			for(i=0;i<nr_w;i++)
			{
				// old way
				// const NUMBER* pw = w + i*n*3 + (ix[px]-1)*3;
				
				// switch back to LIBLINEAR ordering
				const NUMBER* pw = w + (ix[px]-1)*nr_w*3 + i * 3;
				dec_values[i] += (pw[2]*logxvalue + pw[1])*logxvalue + pw[0];
			}
		}
	}

	if(nr_class==2)
		return (dec_values[0]>0) ? label[0] : label[1];
	else
	{
		int dec_max_idx = 0;
		for(i=1;i<nr_class;i++)
		{
			if(dec_values[i] > dec_values[dec_max_idx])
				dec_max_idx = i;
		}
		return label[dec_max_idx];
	}
}

int model::Predict(const int* ix,const NUMBER* vx)
{
	int label=PredictValues(ix, vx, _dec_values);
	return label;
}

NUMBER problem::CrossValidation(const int nr_fold, const NUMBER C,const NUMBER p)
{
	int max_nr_class = 16;
	int nr_class = 0;
	int *label = (int*)malloc((unsigned long)max_nr_class*sizeof(int));

	for(int i=0;i<l;i++)
	{
		int this_label = y[i];
		int j;
		for(j=0;j<nr_class;j++) if(this_label == label[j]) break;
		if(j == nr_class)
		{
			if(nr_class == max_nr_class)
			{
				max_nr_class *= 2;
				label = (int *)realloc(label,max_nr_class*sizeof(int));
			}
			label[nr_class] = this_label;
			++nr_class;
		}
	}
	num_class = nr_class;
	if(labels) delete[] labels;
	labels = new int[nr_class];
	std::copy(label,label+nr_class,labels);
	free(label);
	
	int *fold_start = new int[nr_fold+1];
	int *perm = new int[l];

	for(int i=0;i<l;i++) perm[i]=i;
	for(int i=0;i<l;i++)
	{
		int j = i+rand()%(l-i);
		std::swap(perm[i],perm[j]);
	}
	for(int i=0;i<=nr_fold;i++) fold_start[i]=i*l/nr_fold;

	int* pd = new int[l];
    int* gt = new int[l];
	for(int i=0;i<nr_fold;i++)
	{
		int begin = fold_start[i];
		int end = fold_start[i+1];
		int j,k;
		
		problem subprob;
		subprob.n = n;
		subprob.l = l-(end-begin);
		subprob.indexes = new int*[subprob.l];
		subprob.values = new NUMBER*[subprob.l];
		subprob.y = new int[subprob.l];
		subprob.index_buf = index_buf; // This is required to compute offset for 'logxplus'
		subprob.value_buf = value_buf;
		subprob.allocated = allocated;

		k=0;
		for(j=0;j<begin;j++)
		{
			subprob.indexes[k] = indexes[perm[j]];
			subprob.values[k] = values[perm[j]];
			subprob.y[k] = y[perm[j]];
			++k;
		}
		for(j=end;j<l;j++)
		{
			subprob.indexes[k] = indexes[perm[j]];
			subprob.values[k] = values[perm[j]];
			subprob.y[k] = y[perm[j]];
			++k;
		}
		model submodel;
		subprob.Train(submodel,C,p);
		for(j=begin;j<end;j++)
		{
			gt[j] = FindLabel(y[j]);
			pd[perm[j]] = FindLabel(submodel.PredictValues(indexes[perm[j]],values[perm[j]],submodel._dec_values));
		}
		submodel.~model();
		subprob.index_buf = NULL; // avoiding double free
		subprob.value_buf = NULL;
		subprob.allocated = 0;
		subprob.Clear();
	}
	
	NUMBER acc, accavg;
	ComputeAccuracies(l,num_class,gt,pd,acc,accavg);
	std::cout<<"Overall accuracy = "<<acc<<std::endl;
	std::cout<<"Average accuracy = "<<accavg<<std::endl;
	delete[] pd; delete[] gt;

	delete[] fold_start;
	return acc;
}

int main(int argc, char **argv)
{
	verbose = true;
	if(argc!=3 && argc!=2)
	{
		std::cout<<"For training followed by testing, "<<std::endl;
		std::cout<<"    Usage: ./pmsvm train_filename test_filename"<<std::endl;
		std::cout<<"Or, for 5-fold cross validation,"<<std::endl;
		std::cout<<"    Usage: ./pmsvm cross_validation_file_name"<<std::endl;
		return 0;
	}
	
	// Load the training (argv[1]) and testing (argv[2]) file
	problem probtrain,probtest;
    probtrain.Load(argv[1],1);

	if(argc==3)
	{
		model model_;
		probtrain.Train(model_,0.01,-1);
		probtrain.Clear();
		
		probtest.Load(argv[2],1,model_.nr_feature);
		int* pd = new int[(unsigned long)probtest.GetNumExamples()];
		int* gt = new int[(unsigned long)probtest.GetNumExamples()];
		NUMBER* table = new NUMBER[(unsigned long)probtest.GetNumExamples()*model_.nr_class];

		timeb TimingMilliSeconds;
		ftime(&TimingMilliSeconds);
		for(int i=0;i<probtest.GetNumExamples();i++)
		{
			gt[i] = model_.FindLabel(probtest.GetLabel(i));
			pd[i] = model_.FindLabel(model_.PredictValues(probtest.GetFeatureIndexes(i),probtest.GetFeatureValues(i),table+i*model_.nr_class));
		}
		struct timeb now;
		ftime(&now);
		std::cout<<"Finished in "<<int( (now.time-TimingMilliSeconds.time)*1000+(now.millitm-TimingMilliSeconds.millitm) )<<" msec."<<std::endl;

		NUMBER acc, accavg;
		ComputeAccuracies(probtest.GetNumExamples(),model_.nr_class,gt,pd,acc,accavg);
		std::cout<<"Overall accuracy = "<<acc<<std::endl;
		std::cout<<"Average accuracy = "<<accavg<<std::endl;
		delete[] pd; delete[] gt;
	}
	else
		probtrain.CrossValidation(5,0.01,-1);
	return 0;
}
