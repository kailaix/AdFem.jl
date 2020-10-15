#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include<cmath>

// Signatures for GPU kernels here 


using namespace tensorflow;
#include "BdmTractionTermMfem.h"


REGISTER_OP("BdmTractionTermMfem")
.Input("gd : double")
.Input("edges : int64")
.Output("rhs : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle gd_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &gd_shape));
        shape_inference::ShapeHandle edges_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &edges_shape));

        c->set_output(0, c->Vector(-1));
    return Status::OK();
  });

REGISTER_OP("BdmTractionTermMfemGrad")
.Input("grad_rhs : double")
.Input("rhs : double")
.Input("gd : double")
.Input("edges : int64")
.Output("grad_gd : double")
.Output("grad_edges : int64");

/*-------------------------------------------------------------------------------------*/

class BdmTractionTermMfemOp : public OpKernel {
private:
  
public:
  explicit BdmTractionTermMfemOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(2, context->num_inputs());
    
    
    const Tensor& gd = context->input(0);
    const Tensor& edges = context->input(1);
    
    
    const TensorShape& gd_shape = gd.shape();
    const TensorShape& edges_shape = edges.shape();
    
    
    DCHECK_EQ(gd_shape.dims(), 1);
    DCHECK_EQ(edges_shape.dims(), 1);

    // extra check
        
    // create output shape
    
    TensorShape rhs_shape({-1});
            
    // create output tensor
    
    Tensor* rhs = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, rhs_shape, &rhs));
    
    // get the corresponding Eigen tensors for data access
    
    auto gd_tensor = gd.flat<double>().data();
    auto edges_tensor = edges.flat<int64>().data();
    auto rhs_tensor = rhs->flat<double>().data();   

    // implement your forward function here 

    // TODO:

  }
};
REGISTER_KERNEL_BUILDER(Name("BdmTractionTermMfem").Device(DEVICE_CPU), BdmTractionTermMfemOp);



class BdmTractionTermMfemGradOp : public OpKernel {
private:
  
public:
  explicit BdmTractionTermMfemGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_rhs = context->input(0);
    const Tensor& rhs = context->input(1);
    const Tensor& gd = context->input(2);
    const Tensor& edges = context->input(3);
    
    
    const TensorShape& grad_rhs_shape = grad_rhs.shape();
    const TensorShape& rhs_shape = rhs.shape();
    const TensorShape& gd_shape = gd.shape();
    const TensorShape& edges_shape = edges.shape();
    
    
    DCHECK_EQ(grad_rhs_shape.dims(), 1);
    DCHECK_EQ(rhs_shape.dims(), 1);
    DCHECK_EQ(gd_shape.dims(), 1);
    DCHECK_EQ(edges_shape.dims(), 1);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_gd_shape(gd_shape);
    TensorShape grad_edges_shape(edges_shape);
            
    // create output tensor
    
    Tensor* grad_gd = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_gd_shape, &grad_gd));
    Tensor* grad_edges = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_edges_shape, &grad_edges));
    
    // get the corresponding Eigen tensors for data access
    
    auto gd_tensor = gd.flat<double>().data();
    auto edges_tensor = edges.flat<int64>().data();
    auto grad_rhs_tensor = grad_rhs.flat<double>().data();
    auto rhs_tensor = rhs.flat<double>().data();
    auto grad_gd_tensor = grad_gd->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    
  }
};
REGISTER_KERNEL_BUILDER(Name("BdmTractionTermMfemGrad").Device(DEVICE_CPU), BdmTractionTermMfemGradOp);


/***************************************************************************************
**********************            GPU Operator            ******************************
***************************************************************************************/


#ifdef GOOGLE_CUDA
class BdmTractionTermMfemOpGPU : public OpKernel {
private:
  
public:
  explicit BdmTractionTermMfemOpGPU(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(2, context->num_inputs());
    
    
    const Tensor& gd = context->input(0);
    const Tensor& edges = context->input(1);
    
    
    const TensorShape& gd_shape = gd.shape();
    const TensorShape& edges_shape = edges.shape();
    
    
    DCHECK_EQ(gd_shape.dims(), 1);
    DCHECK_EQ(edges_shape.dims(), 1);

    // extra check
        
    // create output shape
    
    TensorShape rhs_shape({-1});
            
    // create output tensor
    
    Tensor* rhs = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, rhs_shape, &rhs));
    
    // get the corresponding Eigen tensors for data access
    
    auto gd_tensor = gd.flat<double>().data();
    auto edges_tensor = edges.flat<int64>().data();
    auto rhs_tensor = rhs->flat<double>().data();   

    // implement your forward function here 

    // TODO:

  }
};
REGISTER_KERNEL_BUILDER(Name("BdmTractionTermMfem").Device(DEVICE_GPU), BdmTractionTermMfemOpGPU);

class BdmTractionTermMfemGradOpGPU : public OpKernel {
private:
  
public:
  explicit BdmTractionTermMfemGradOpGPU(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_rhs = context->input(0);
    const Tensor& rhs = context->input(1);
    const Tensor& gd = context->input(2);
    const Tensor& edges = context->input(3);
    
    
    const TensorShape& grad_rhs_shape = grad_rhs.shape();
    const TensorShape& rhs_shape = rhs.shape();
    const TensorShape& gd_shape = gd.shape();
    const TensorShape& edges_shape = edges.shape();
    
    
    DCHECK_EQ(grad_rhs_shape.dims(), 1);
    DCHECK_EQ(rhs_shape.dims(), 1);
    DCHECK_EQ(gd_shape.dims(), 1);
    DCHECK_EQ(edges_shape.dims(), 1);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_gd_shape(gd_shape);
    TensorShape grad_edges_shape(edges_shape);
            
    // create output tensor
    
    Tensor* grad_gd = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_gd_shape, &grad_gd));
    Tensor* grad_edges = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_edges_shape, &grad_edges));
    
    // get the corresponding Eigen tensors for data access
    
    auto gd_tensor = gd.flat<double>().data();
    auto edges_tensor = edges.flat<int64>().data();
    auto grad_rhs_tensor = grad_rhs.flat<double>().data();
    auto rhs_tensor = rhs.flat<double>().data();
    auto grad_gd_tensor = grad_gd->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    
  }
};
REGISTER_KERNEL_BUILDER(Name("BdmTractionTermMfemGrad").Device(DEVICE_GPU), BdmTractionTermMfemGradOpGPU);

#endif