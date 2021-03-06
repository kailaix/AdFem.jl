#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include<cmath>

// Signatures for GPU kernels here 


using namespace tensorflow;
#include "ComputeFemMassMatrixMfemT.h"


REGISTER_OP("ComputeFemMassMatrixMfemT")
.Input("rho : double")
.Output("indices : int64")
.Output("vv : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle rho_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &rho_shape));

        c->set_output(0, c->Matrix(-1,2));
        c->set_output(1, c->Vector(-1));
    return Status::OK();
  });

REGISTER_OP("ComputeFemMassMatrixMfemTGrad")
.Input("grad_vv : double")
.Input("indices : int64")
.Input("vv : double")
.Input("rho : double")
.Output("grad_rho : double");

/*-------------------------------------------------------------------------------------*/

class ComputeFemMassMatrixMfemTOp : public OpKernel {
private:
  
public:
  explicit ComputeFemMassMatrixMfemTOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(1, context->num_inputs());
    
    
    const Tensor& rho = context->input(0);
    
    
    const TensorShape& rho_shape = rho.shape();
    
    
    DCHECK_EQ(rho_shape.dims(), 1);

    // extra check
        
    // create output shape
    
    TensorShape indices_shape({mmesh3.nelem * mmesh3.elem_ndof * mmesh3.elem_ndof,2});
    TensorShape vv_shape({mmesh3.nelem * mmesh3.elem_ndof * mmesh3.elem_ndof});
            
    // create output tensor
    
    Tensor* indices = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, indices_shape, &indices));
    Tensor* vv = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, vv_shape, &vv));
    
    // get the corresponding Eigen tensors for data access
    
    auto rho_tensor = rho.flat<double>().data();
    auto indices_tensor = indices->flat<int64>().data();
    auto vv_tensor = vv->flat<double>().data();   

    // implement your forward function here 

    // TODO:
    MFEM::ComputeFemMassMatrixMfemT_forward(indices_tensor, vv_tensor, rho_tensor);

  }
};
REGISTER_KERNEL_BUILDER(Name("ComputeFemMassMatrixMfemT").Device(DEVICE_CPU), ComputeFemMassMatrixMfemTOp);



class ComputeFemMassMatrixMfemTGradOp : public OpKernel {
private:
  
public:
  explicit ComputeFemMassMatrixMfemTGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_vv = context->input(0);
    const Tensor& indices = context->input(1);
    const Tensor& vv = context->input(2);
    const Tensor& rho = context->input(3);
    
    
    const TensorShape& grad_vv_shape = grad_vv.shape();
    const TensorShape& indices_shape = indices.shape();
    const TensorShape& vv_shape = vv.shape();
    const TensorShape& rho_shape = rho.shape();
    
    
    DCHECK_EQ(grad_vv_shape.dims(), 1);
    DCHECK_EQ(indices_shape.dims(), 2);
    DCHECK_EQ(vv_shape.dims(), 1);
    DCHECK_EQ(rho_shape.dims(), 1);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_rho_shape(rho_shape);
            
    // create output tensor
    
    Tensor* grad_rho = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_rho_shape, &grad_rho));
    
    // get the corresponding Eigen tensors for data access
    
    auto rho_tensor = rho.flat<double>().data();
    auto grad_vv_tensor = grad_vv.flat<double>().data();
    auto indices_tensor = indices.flat<int64>().data();
    auto vv_tensor = vv.flat<double>().data();
    auto grad_rho_tensor = grad_rho->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    
  }
};
REGISTER_KERNEL_BUILDER(Name("ComputeFemMassMatrixMfemTGrad").Device(DEVICE_CPU), ComputeFemMassMatrixMfemTGradOp);


/***************************************************************************************
**********************            GPU Operator            ******************************
***************************************************************************************/


#ifdef GOOGLE_CUDA

REGISTER_OP("ComputeFemMassMatrixMfemTGpu")
.Input("rho : double")
.Output("indices : int64")
.Output("vv : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle rho_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &rho_shape));

        c->set_output(0, c->Matrix(-1,2));
        c->set_output(1, c->Vector(-1));
    return Status::OK();
  });

REGISTER_OP("ComputeFemMassMatrixMfemTGpuGrad")
.Input("grad_vv : double")
.Input("indices : int64")
.Input("vv : double")
.Input("rho : double")
.Output("grad_rho : double");

class ComputeFemMassMatrixMfemTOpGPU : public OpKernel {
private:
  
public:
  explicit ComputeFemMassMatrixMfemTOpGPU(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(1, context->num_inputs());
    
    
    const Tensor& rho = context->input(0);
    
    
    const TensorShape& rho_shape = rho.shape();
    
    
    DCHECK_EQ(rho_shape.dims(), 1);

    // extra check
        
    // create output shape
    
    TensorShape indices_shape({-1,2});
    TensorShape vv_shape({-1});
            
    // create output tensor
    
    Tensor* indices = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, indices_shape, &indices));
    Tensor* vv = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, vv_shape, &vv));
    
    // get the corresponding Eigen tensors for data access
    
    auto rho_tensor = rho.flat<double>().data();
    auto indices_tensor = indices->flat<int64>().data();
    auto vv_tensor = vv->flat<double>().data();   

    // implement your forward function here 

    // TODO:

  }
};
REGISTER_KERNEL_BUILDER(Name("ComputeFemMassMatrixMfemTGpu").Device(DEVICE_GPU), ComputeFemMassMatrixMfemTOpGPU);

class ComputeFemMassMatrixMfemTGradOpGPU : public OpKernel {
private:
  
public:
  explicit ComputeFemMassMatrixMfemTGradOpGPU(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_vv = context->input(0);
    const Tensor& indices = context->input(1);
    const Tensor& vv = context->input(2);
    const Tensor& rho = context->input(3);
    
    
    const TensorShape& grad_vv_shape = grad_vv.shape();
    const TensorShape& indices_shape = indices.shape();
    const TensorShape& vv_shape = vv.shape();
    const TensorShape& rho_shape = rho.shape();
    
    
    DCHECK_EQ(grad_vv_shape.dims(), 1);
    DCHECK_EQ(indices_shape.dims(), 2);
    DCHECK_EQ(vv_shape.dims(), 1);
    DCHECK_EQ(rho_shape.dims(), 1);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_rho_shape(rho_shape);
            
    // create output tensor
    
    Tensor* grad_rho = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_rho_shape, &grad_rho));
    
    // get the corresponding Eigen tensors for data access
    
    auto rho_tensor = rho.flat<double>().data();
    auto grad_vv_tensor = grad_vv.flat<double>().data();
    auto indices_tensor = indices.flat<int64>().data();
    auto vv_tensor = vv.flat<double>().data();
    auto grad_rho_tensor = grad_rho->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    
  }
};
REGISTER_KERNEL_BUILDER(Name("ComputeFemMassMatrixMfemTGpuGrad").Device(DEVICE_GPU), ComputeFemMassMatrixMfemTGradOpGPU);

#endif