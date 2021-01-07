#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include<cmath>

// Signatures for GPU kernels here 


using namespace tensorflow;
#include "ComputePmlTerm.h"


REGISTER_OP("ComputePmlTerm")
.Input("u : double")
.Input("betap : double")
.Input("c : double")
.Input("nv : double")
.Output("k1 : double")
.Output("k2 : double")
.Output("k3 : double")
.Output("k4 : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle u_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &u_shape));
        shape_inference::ShapeHandle betap_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &betap_shape));
        shape_inference::ShapeHandle c_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &c_shape));
        shape_inference::ShapeHandle nv_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 2, &nv_shape));

        c->set_output(0, c->Vector(-1));
        c->set_output(1, c->Vector(-1));
        c->set_output(2, c->Vector(-1));
        c->set_output(3, c->Vector(-1));
    return Status::OK();
  });

REGISTER_OP("ComputePmlTermGrad")
.Input("grad_k1 : double")
.Input("grad_k2 : double")
.Input("grad_k3 : double")
.Input("grad_k4 : double")
.Input("k1 : double")
.Input("k2 : double")
.Input("k3 : double")
.Input("k4 : double")
.Input("u : double")
.Input("betap : double")
.Input("c : double")
.Input("nv : double")
.Output("grad_u : double")
.Output("grad_betap : double")
.Output("grad_c : double")
.Output("grad_nv : double");

/*-------------------------------------------------------------------------------------*/

class ComputePmlTermOp : public OpKernel {
private:
  
public:
  explicit ComputePmlTermOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(4, context->num_inputs());
    
    
    const Tensor& u = context->input(0);
    const Tensor& betap = context->input(1);
    const Tensor& c = context->input(2);
    const Tensor& nv = context->input(3);
    
    
    const TensorShape& u_shape = u.shape();
    const TensorShape& betap_shape = betap.shape();
    const TensorShape& c_shape = c.shape();
    const TensorShape& nv_shape = nv.shape();
    
    
    DCHECK_EQ(u_shape.dims(), 1);
    DCHECK_EQ(betap_shape.dims(), 1);
    DCHECK_EQ(c_shape.dims(), 1);
    DCHECK_EQ(nv_shape.dims(), 2);

    // extra check
        
    // create output shape
    
    TensorShape k1_shape({mmesh.ndof});
    TensorShape k2_shape({mmesh.ndof});
    TensorShape k3_shape({mmesh.ndof});
    TensorShape k4_shape({mmesh.ndof});
            
    // create output tensor
    
    Tensor* k1 = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, k1_shape, &k1));
    Tensor* k2 = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, k2_shape, &k2));
    Tensor* k3 = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, k3_shape, &k3));
    Tensor* k4 = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(3, k4_shape, &k4));
    
    // get the corresponding Eigen tensors for data access
    
    auto u_tensor = u.flat<double>().data();
    auto betap_tensor = betap.flat<double>().data();
    auto c_tensor = c.flat<double>().data();
    auto nv_tensor = nv.flat<double>().data();
    auto k1_tensor = k1->flat<double>().data();
    auto k2_tensor = k2->flat<double>().data();
    auto k3_tensor = k3->flat<double>().data();
    auto k4_tensor = k4->flat<double>().data();   

    // implement your forward function here 

    // TODO:
    k1->flat<double>().setZero();
    k2->flat<double>().setZero();
    k3->flat<double>().setZero();
    k4->flat<double>().setZero();
    MFEM::ComputePmlTermForward(k1_tensor, k2_tensor, k3_tensor, k4_tensor, 
              u_tensor, betap_tensor, c_tensor, nv_tensor);

  }
};
REGISTER_KERNEL_BUILDER(Name("ComputePmlTerm").Device(DEVICE_CPU), ComputePmlTermOp);



class ComputePmlTermGradOp : public OpKernel {
private:
  
public:
  explicit ComputePmlTermGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_k1 = context->input(0);
    const Tensor& grad_k2 = context->input(1);
    const Tensor& grad_k3 = context->input(2);
    const Tensor& grad_k4 = context->input(3);
    const Tensor& k1 = context->input(4);
    const Tensor& k2 = context->input(5);
    const Tensor& k3 = context->input(6);
    const Tensor& k4 = context->input(7);
    const Tensor& u = context->input(8);
    const Tensor& betap = context->input(9);
    const Tensor& c = context->input(10);
    const Tensor& nv = context->input(11);
    
    
    const TensorShape& grad_k1_shape = grad_k1.shape();
    const TensorShape& grad_k2_shape = grad_k2.shape();
    const TensorShape& grad_k3_shape = grad_k3.shape();
    const TensorShape& grad_k4_shape = grad_k4.shape();
    const TensorShape& k1_shape = k1.shape();
    const TensorShape& k2_shape = k2.shape();
    const TensorShape& k3_shape = k3.shape();
    const TensorShape& k4_shape = k4.shape();
    const TensorShape& u_shape = u.shape();
    const TensorShape& betap_shape = betap.shape();
    const TensorShape& c_shape = c.shape();
    const TensorShape& nv_shape = nv.shape();
    
    
    DCHECK_EQ(grad_k1_shape.dims(), 1);
    DCHECK_EQ(grad_k2_shape.dims(), 1);
    DCHECK_EQ(grad_k3_shape.dims(), 1);
    DCHECK_EQ(grad_k4_shape.dims(), 1);
    DCHECK_EQ(k1_shape.dims(), 1);
    DCHECK_EQ(k2_shape.dims(), 1);
    DCHECK_EQ(k3_shape.dims(), 1);
    DCHECK_EQ(k4_shape.dims(), 1);
    DCHECK_EQ(u_shape.dims(), 1);
    DCHECK_EQ(betap_shape.dims(), 1);
    DCHECK_EQ(c_shape.dims(), 1);
    DCHECK_EQ(nv_shape.dims(), 2);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_u_shape(u_shape);
    TensorShape grad_betap_shape(betap_shape);
    TensorShape grad_c_shape(c_shape);
    TensorShape grad_nv_shape(nv_shape);
            
    // create output tensor
    
    Tensor* grad_u = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_u_shape, &grad_u));
    Tensor* grad_betap = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_betap_shape, &grad_betap));
    Tensor* grad_c = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_c_shape, &grad_c));
    Tensor* grad_nv = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(3, grad_nv_shape, &grad_nv));
    
    // get the corresponding Eigen tensors for data access
    
    auto u_tensor = u.flat<double>().data();
    auto betap_tensor = betap.flat<double>().data();
    auto c_tensor = c.flat<double>().data();
    auto nv_tensor = nv.flat<double>().data();
    auto grad_k1_tensor = grad_k1.flat<double>().data();
    auto grad_k2_tensor = grad_k2.flat<double>().data();
    auto grad_k3_tensor = grad_k3.flat<double>().data();
    auto grad_k4_tensor = grad_k4.flat<double>().data();
    auto k1_tensor = k1.flat<double>().data();
    auto k2_tensor = k2.flat<double>().data();
    auto k3_tensor = k3.flat<double>().data();
    auto k4_tensor = k4.flat<double>().data();
    auto grad_u_tensor = grad_u->flat<double>().data();
    auto grad_betap_tensor = grad_betap->flat<double>().data();
    auto grad_c_tensor = grad_c->flat<double>().data();
    auto grad_nv_tensor = grad_nv->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    
  }
};
REGISTER_KERNEL_BUILDER(Name("ComputePmlTermGrad").Device(DEVICE_CPU), ComputePmlTermGradOp);


/***************************************************************************************
**********************            GPU Operator            ******************************
***************************************************************************************/


#ifdef GOOGLE_CUDA

REGISTER_OP("ComputePmlTermGpu")
.Input("u : double")
.Input("betap : double")
.Input("c : double")
.Input("nv : double")
.Output("k1 : double")
.Output("k2 : double")
.Output("k3 : double")
.Output("k4 : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle u_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &u_shape));
        shape_inference::ShapeHandle betap_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &betap_shape));
        shape_inference::ShapeHandle c_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &c_shape));
        shape_inference::ShapeHandle nv_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 2, &nv_shape));

        c->set_output(0, c->Vector(-1));
        c->set_output(1, c->Vector(-1));
        c->set_output(2, c->Vector(-1));
        c->set_output(3, c->Vector(-1));
    return Status::OK();
  });

REGISTER_OP("ComputePmlTermGpuGrad")
.Input("grad_k1 : double")
.Input("grad_k2 : double")
.Input("grad_k3 : double")
.Input("grad_k4 : double")
.Input("k1 : double")
.Input("k2 : double")
.Input("k3 : double")
.Input("k4 : double")
.Input("u : double")
.Input("betap : double")
.Input("c : double")
.Input("nv : double")
.Output("grad_u : double")
.Output("grad_betap : double")
.Output("grad_c : double")
.Output("grad_nv : double");

class ComputePmlTermOpGPU : public OpKernel {
private:
  
public:
  explicit ComputePmlTermOpGPU(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(4, context->num_inputs());
    
    
    const Tensor& u = context->input(0);
    const Tensor& betap = context->input(1);
    const Tensor& c = context->input(2);
    const Tensor& nv = context->input(3);
    
    
    const TensorShape& u_shape = u.shape();
    const TensorShape& betap_shape = betap.shape();
    const TensorShape& c_shape = c.shape();
    const TensorShape& nv_shape = nv.shape();
    
    
    DCHECK_EQ(u_shape.dims(), 1);
    DCHECK_EQ(betap_shape.dims(), 1);
    DCHECK_EQ(c_shape.dims(), 1);
    DCHECK_EQ(nv_shape.dims(), 2);

    // extra check
        
    // create output shape
    
    TensorShape k1_shape({-1});
    TensorShape k2_shape({-1});
    TensorShape k3_shape({-1});
    TensorShape k4_shape({-1});
            
    // create output tensor
    
    Tensor* k1 = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, k1_shape, &k1));
    Tensor* k2 = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, k2_shape, &k2));
    Tensor* k3 = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, k3_shape, &k3));
    Tensor* k4 = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(3, k4_shape, &k4));
    
    // get the corresponding Eigen tensors for data access
    
    auto u_tensor = u.flat<double>().data();
    auto betap_tensor = betap.flat<double>().data();
    auto c_tensor = c.flat<double>().data();
    auto nv_tensor = nv.flat<double>().data();
    auto k1_tensor = k1->flat<double>().data();
    auto k2_tensor = k2->flat<double>().data();
    auto k3_tensor = k3->flat<double>().data();
    auto k4_tensor = k4->flat<double>().data();   

    // implement your forward function here 

    // TODO:

  }
};
REGISTER_KERNEL_BUILDER(Name("ComputePmlTermGpu").Device(DEVICE_GPU), ComputePmlTermOpGPU);

class ComputePmlTermGradOpGPU : public OpKernel {
private:
  
public:
  explicit ComputePmlTermGradOpGPU(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_k1 = context->input(0);
    const Tensor& grad_k2 = context->input(1);
    const Tensor& grad_k3 = context->input(2);
    const Tensor& grad_k4 = context->input(3);
    const Tensor& k1 = context->input(4);
    const Tensor& k2 = context->input(5);
    const Tensor& k3 = context->input(6);
    const Tensor& k4 = context->input(7);
    const Tensor& u = context->input(8);
    const Tensor& betap = context->input(9);
    const Tensor& c = context->input(10);
    const Tensor& nv = context->input(11);
    
    
    const TensorShape& grad_k1_shape = grad_k1.shape();
    const TensorShape& grad_k2_shape = grad_k2.shape();
    const TensorShape& grad_k3_shape = grad_k3.shape();
    const TensorShape& grad_k4_shape = grad_k4.shape();
    const TensorShape& k1_shape = k1.shape();
    const TensorShape& k2_shape = k2.shape();
    const TensorShape& k3_shape = k3.shape();
    const TensorShape& k4_shape = k4.shape();
    const TensorShape& u_shape = u.shape();
    const TensorShape& betap_shape = betap.shape();
    const TensorShape& c_shape = c.shape();
    const TensorShape& nv_shape = nv.shape();
    
    
    DCHECK_EQ(grad_k1_shape.dims(), 1);
    DCHECK_EQ(grad_k2_shape.dims(), 1);
    DCHECK_EQ(grad_k3_shape.dims(), 1);
    DCHECK_EQ(grad_k4_shape.dims(), 1);
    DCHECK_EQ(k1_shape.dims(), 1);
    DCHECK_EQ(k2_shape.dims(), 1);
    DCHECK_EQ(k3_shape.dims(), 1);
    DCHECK_EQ(k4_shape.dims(), 1);
    DCHECK_EQ(u_shape.dims(), 1);
    DCHECK_EQ(betap_shape.dims(), 1);
    DCHECK_EQ(c_shape.dims(), 1);
    DCHECK_EQ(nv_shape.dims(), 2);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_u_shape(u_shape);
    TensorShape grad_betap_shape(betap_shape);
    TensorShape grad_c_shape(c_shape);
    TensorShape grad_nv_shape(nv_shape);
            
    // create output tensor
    
    Tensor* grad_u = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_u_shape, &grad_u));
    Tensor* grad_betap = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_betap_shape, &grad_betap));
    Tensor* grad_c = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_c_shape, &grad_c));
    Tensor* grad_nv = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(3, grad_nv_shape, &grad_nv));
    
    // get the corresponding Eigen tensors for data access
    
    auto u_tensor = u.flat<double>().data();
    auto betap_tensor = betap.flat<double>().data();
    auto c_tensor = c.flat<double>().data();
    auto nv_tensor = nv.flat<double>().data();
    auto grad_k1_tensor = grad_k1.flat<double>().data();
    auto grad_k2_tensor = grad_k2.flat<double>().data();
    auto grad_k3_tensor = grad_k3.flat<double>().data();
    auto grad_k4_tensor = grad_k4.flat<double>().data();
    auto k1_tensor = k1.flat<double>().data();
    auto k2_tensor = k2.flat<double>().data();
    auto k3_tensor = k3.flat<double>().data();
    auto k4_tensor = k4.flat<double>().data();
    auto grad_u_tensor = grad_u->flat<double>().data();
    auto grad_betap_tensor = grad_betap->flat<double>().data();
    auto grad_c_tensor = grad_c->flat<double>().data();
    auto grad_nv_tensor = grad_nv->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    
  }
};
REGISTER_KERNEL_BUILDER(Name("ComputePmlTermGpuGrad").Device(DEVICE_GPU), ComputePmlTermGradOpGPU);

#endif