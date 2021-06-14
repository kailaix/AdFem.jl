#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include<cmath>

// Signatures for GPU kernels here 


using namespace tensorflow;
#include "EvalNormalAndShearStressOnBoundaryEdge.h"


REGISTER_OP("EvalNormalAndShearStressOnBoundaryEdge")
.Input("sigma : double")
.Input("n : double")
.Output("sn : double")
.Output("st : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle sigma_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &sigma_shape));
        shape_inference::ShapeHandle n_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &n_shape));

        c->set_output(0, c->Vector(-1));
        c->set_output(1, c->Vector(-1));
    return Status::OK();
  });

REGISTER_OP("EvalNormalAndShearStressOnBoundaryEdgeGrad")
.Input("grad_sn : double")
.Input("grad_st : double")
.Input("sn : double")
.Input("st : double")
.Input("sigma : double")
.Input("n : double")
.Output("grad_sigma : double")
.Output("grad_n : double");

/*-------------------------------------------------------------------------------------*/

class EvalNormalAndShearStressOnBoundaryEdgeOp : public OpKernel {
private:
  
public:
  explicit EvalNormalAndShearStressOnBoundaryEdgeOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(2, context->num_inputs());
    
    
    const Tensor& sigma = context->input(0);
    const Tensor& n = context->input(1);
    
    
    const TensorShape& sigma_shape = sigma.shape();
    const TensorShape& n_shape = n.shape();
    
    
    DCHECK_EQ(sigma_shape.dims(), 2);
    DCHECK_EQ(n_shape.dims(), 2);

    // extra check
        
    // create output shape
    int N = sigma_shape.dim_size(0);
    TensorShape sn_shape({N});
    TensorShape st_shape({N});
            
    // create output tensor
    
    Tensor* sn = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, sn_shape, &sn));
    Tensor* st = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, st_shape, &st));
    
    // get the corresponding Eigen tensors for data access
    
    auto sigma_tensor = sigma.flat<double>().data();
    auto n_tensor = n.flat<double>().data();
    auto sn_tensor = sn->flat<double>().data();
    auto st_tensor = st->flat<double>().data();   

    // implement your forward function here 

    // TODO:
    MFEM::EvalNormalAndShearStressOnBoundaryEdgeForward(sn_tensor, st_tensor, sigma_tensor, n_tensor, N);

  }
};
REGISTER_KERNEL_BUILDER(Name("EvalNormalAndShearStressOnBoundaryEdge").Device(DEVICE_CPU), EvalNormalAndShearStressOnBoundaryEdgeOp);



class EvalNormalAndShearStressOnBoundaryEdgeGradOp : public OpKernel {
private:
  
public:
  explicit EvalNormalAndShearStressOnBoundaryEdgeGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_sn = context->input(0);
    const Tensor& grad_st = context->input(1);
    const Tensor& sn = context->input(2);
    const Tensor& st = context->input(3);
    const Tensor& sigma = context->input(4);
    const Tensor& n = context->input(5);
    
    
    const TensorShape& grad_sn_shape = grad_sn.shape();
    const TensorShape& grad_st_shape = grad_st.shape();
    const TensorShape& sn_shape = sn.shape();
    const TensorShape& st_shape = st.shape();
    const TensorShape& sigma_shape = sigma.shape();
    const TensorShape& n_shape = n.shape();
    
    
    DCHECK_EQ(grad_sn_shape.dims(), 1);
    DCHECK_EQ(grad_st_shape.dims(), 1);
    DCHECK_EQ(sn_shape.dims(), 1);
    DCHECK_EQ(st_shape.dims(), 1);
    DCHECK_EQ(sigma_shape.dims(), 2);
    DCHECK_EQ(n_shape.dims(), 2);

    // extra check
    int N = sigma_shape.dim_size(0);
    
    TensorShape grad_sigma_shape(sigma_shape);
    TensorShape grad_n_shape(n_shape);
            
    // create output tensor
    
    Tensor* grad_sigma = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_sigma_shape, &grad_sigma));
    Tensor* grad_n = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_n_shape, &grad_n));
    
    // get the corresponding Eigen tensors for data access
    
    auto sigma_tensor = sigma.flat<double>().data();
    auto n_tensor = n.flat<double>().data();
    auto grad_sn_tensor = grad_sn.flat<double>().data();
    auto grad_st_tensor = grad_st.flat<double>().data();
    auto sn_tensor = sn.flat<double>().data();
    auto st_tensor = st.flat<double>().data();
    auto grad_sigma_tensor = grad_sigma->flat<double>().data();
    auto grad_n_tensor = grad_n->flat<double>().data();   

    // implement your backward function here 


    MFEM::EvalNormalAndShearStressOnBoundaryEdgeBackward(
      grad_sigma_tensor, grad_sn_tensor, grad_st_tensor, 
      sn_tensor, st_tensor, sigma_tensor, n_tensor, N
      );

  }
};
REGISTER_KERNEL_BUILDER(Name("EvalNormalAndShearStressOnBoundaryEdgeGrad").Device(DEVICE_CPU), EvalNormalAndShearStressOnBoundaryEdgeGradOp);


/***************************************************************************************
**********************            GPU Operator            ******************************
***************************************************************************************/


#ifdef GOOGLE_CUDA

REGISTER_OP("EvalNormalAndShearStressOnBoundaryEdgeGpu")
.Input("sigma : double")
.Input("n : double")
.Output("sn : double")
.Output("st : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle sigma_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &sigma_shape));
        shape_inference::ShapeHandle n_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &n_shape));

        c->set_output(0, c->Vector(-1));
        c->set_output(1, c->Vector(-1));
    return Status::OK();
  });

REGISTER_OP("EvalNormalAndShearStressOnBoundaryEdgeGpuGrad")
.Input("grad_sn : double")
.Input("grad_st : double")
.Input("sn : double")
.Input("st : double")
.Input("sigma : double")
.Input("n : double")
.Output("grad_sigma : double")
.Output("grad_n : double");

class EvalNormalAndShearStressOnBoundaryEdgeOpGPU : public OpKernel {
private:
  
public:
  explicit EvalNormalAndShearStressOnBoundaryEdgeOpGPU(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(2, context->num_inputs());
    
    
    const Tensor& sigma = context->input(0);
    const Tensor& n = context->input(1);
    
    
    const TensorShape& sigma_shape = sigma.shape();
    const TensorShape& n_shape = n.shape();
    
    
    DCHECK_EQ(sigma_shape.dims(), 2);
    DCHECK_EQ(n_shape.dims(), 2);

    // extra check
        
    // create output shape
    
    TensorShape sn_shape({-1});
    TensorShape st_shape({-1});
            
    // create output tensor
    
    Tensor* sn = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, sn_shape, &sn));
    Tensor* st = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, st_shape, &st));
    
    // get the corresponding Eigen tensors for data access
    
    auto sigma_tensor = sigma.flat<double>().data();
    auto n_tensor = n.flat<double>().data();
    auto sn_tensor = sn->flat<double>().data();
    auto st_tensor = st->flat<double>().data();   

    // implement your forward function here 

    // TODO:

  }
};
REGISTER_KERNEL_BUILDER(Name("EvalNormalAndShearStressOnBoundaryEdgeGpu").Device(DEVICE_GPU), EvalNormalAndShearStressOnBoundaryEdgeOpGPU);

class EvalNormalAndShearStressOnBoundaryEdgeGradOpGPU : public OpKernel {
private:
  
public:
  explicit EvalNormalAndShearStressOnBoundaryEdgeGradOpGPU(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_sn = context->input(0);
    const Tensor& grad_st = context->input(1);
    const Tensor& sn = context->input(2);
    const Tensor& st = context->input(3);
    const Tensor& sigma = context->input(4);
    const Tensor& n = context->input(5);
    
    
    const TensorShape& grad_sn_shape = grad_sn.shape();
    const TensorShape& grad_st_shape = grad_st.shape();
    const TensorShape& sn_shape = sn.shape();
    const TensorShape& st_shape = st.shape();
    const TensorShape& sigma_shape = sigma.shape();
    const TensorShape& n_shape = n.shape();
    
    
    DCHECK_EQ(grad_sn_shape.dims(), 1);
    DCHECK_EQ(grad_st_shape.dims(), 1);
    DCHECK_EQ(sn_shape.dims(), 1);
    DCHECK_EQ(st_shape.dims(), 1);
    DCHECK_EQ(sigma_shape.dims(), 2);
    DCHECK_EQ(n_shape.dims(), 2);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_sigma_shape(sigma_shape);
    TensorShape grad_n_shape(n_shape);
            
    // create output tensor
    
    Tensor* grad_sigma = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_sigma_shape, &grad_sigma));
    Tensor* grad_n = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_n_shape, &grad_n));
    
    // get the corresponding Eigen tensors for data access
    
    auto sigma_tensor = sigma.flat<double>().data();
    auto n_tensor = n.flat<double>().data();
    auto grad_sn_tensor = grad_sn.flat<double>().data();
    auto grad_st_tensor = grad_st.flat<double>().data();
    auto sn_tensor = sn.flat<double>().data();
    auto st_tensor = st.flat<double>().data();
    auto grad_sigma_tensor = grad_sigma->flat<double>().data();
    auto grad_n_tensor = grad_n->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    
  }
};
REGISTER_KERNEL_BUILDER(Name("EvalNormalAndShearStressOnBoundaryEdgeGpuGrad").Device(DEVICE_GPU), EvalNormalAndShearStressOnBoundaryEdgeGradOpGPU);

#endif