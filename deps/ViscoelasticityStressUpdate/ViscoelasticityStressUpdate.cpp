#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include<cmath>

// Signatures for GPU kernels here 


using namespace tensorflow;
#include "ViscoelasticityStressUpdate.h"


REGISTER_OP("ViscoelasticityStressUpdate")
.Input("mu : double")
.Input("lambda : double")
.Input("inveta : double")
.Input("dt : double")
.Input("epsilon1 : double")
.Input("epsilon2 : double")
.Input("sigma1 : double")
.Output("sigma2 : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle mu_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &mu_shape));
        shape_inference::ShapeHandle lambda_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &lambda_shape));
        shape_inference::ShapeHandle inveta_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &inveta_shape));
        shape_inference::ShapeHandle dt_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &dt_shape));
        shape_inference::ShapeHandle epsilon1_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 2, &epsilon1_shape));
        shape_inference::ShapeHandle epsilon2_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 2, &epsilon2_shape));
        shape_inference::ShapeHandle sigma1_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(6), 2, &sigma1_shape));

        c->set_output(0, c->Matrix(-1,3));
    return Status::OK();
  });

REGISTER_OP("ViscoelasticityStressUpdateGrad")
.Input("grad_sigma2 : double")
.Input("sigma2 : double")
.Input("mu : double")
.Input("lambda : double")
.Input("inveta : double")
.Input("dt : double")
.Input("epsilon1 : double")
.Input("epsilon2 : double")
.Input("sigma1 : double")
.Output("grad_mu : double")
.Output("grad_lambda : double")
.Output("grad_inveta : double")
.Output("grad_dt : double")
.Output("grad_epsilon1 : double")
.Output("grad_epsilon2 : double")
.Output("grad_sigma1 : double");

/*-------------------------------------------------------------------------------------*/

class ViscoelasticityStressUpdateOp : public OpKernel {
private:
  
public:
  explicit ViscoelasticityStressUpdateOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(7, context->num_inputs());
    
    
    const Tensor& mu = context->input(0);
    const Tensor& lambda = context->input(1);
    const Tensor& inveta = context->input(2);
    const Tensor& dt = context->input(3);
    const Tensor& epsilon1 = context->input(4);
    const Tensor& epsilon2 = context->input(5);
    const Tensor& sigma1 = context->input(6);
    
    
    const TensorShape& mu_shape = mu.shape();
    const TensorShape& lambda_shape = lambda.shape();
    const TensorShape& inveta_shape = inveta.shape();
    const TensorShape& dt_shape = dt.shape();
    const TensorShape& epsilon1_shape = epsilon1.shape();
    const TensorShape& epsilon2_shape = epsilon2.shape();
    const TensorShape& sigma1_shape = sigma1.shape();
    
    
    DCHECK_EQ(mu_shape.dims(), 1);
    DCHECK_EQ(lambda_shape.dims(), 1);
    DCHECK_EQ(inveta_shape.dims(), 1);
    DCHECK_EQ(dt_shape.dims(), 0);
    DCHECK_EQ(epsilon1_shape.dims(), 2);
    DCHECK_EQ(epsilon2_shape.dims(), 2);
    DCHECK_EQ(sigma1_shape.dims(), 2);

    // extra check
        
    // create output shape
    int ng = mu_shape.dim_size(0);
    TensorShape sigma2_shape({ng,3});
            
    // create output tensor
    
    Tensor* sigma2 = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, sigma2_shape, &sigma2));
    
    // get the corresponding Eigen tensors for data access
    
    auto mu_tensor = mu.flat<double>().data();
    auto lambda_tensor = lambda.flat<double>().data();
    auto inveta_tensor = inveta.flat<double>().data();
    auto dt_tensor = dt.flat<double>().data();
    auto epsilon1_tensor = epsilon1.flat<double>().data();
    auto epsilon2_tensor = epsilon2.flat<double>().data();
    auto sigma1_tensor = sigma1.flat<double>().data();
    auto sigma2_tensor = sigma2->flat<double>().data();   

    // implement your forward function here 

    // TODO:
    ViscoelasticityStressUpdateForward(sigma2_tensor, epsilon1_tensor, epsilon2_tensor, sigma1_tensor, 
          mu_tensor, inveta_tensor, lambda_tensor, *dt_tensor, 0, ng);

  }
};
REGISTER_KERNEL_BUILDER(Name("ViscoelasticityStressUpdate").Device(DEVICE_CPU), ViscoelasticityStressUpdateOp);



class ViscoelasticityStressUpdateGradOp : public OpKernel {
private:
  
public:
  explicit ViscoelasticityStressUpdateGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_sigma2 = context->input(0);
    const Tensor& sigma2 = context->input(1);
    const Tensor& mu = context->input(2);
    const Tensor& lambda = context->input(3);
    const Tensor& inveta = context->input(4);
    const Tensor& dt = context->input(5);
    const Tensor& epsilon1 = context->input(6);
    const Tensor& epsilon2 = context->input(7);
    const Tensor& sigma1 = context->input(8);
    
    
    const TensorShape& grad_sigma2_shape = grad_sigma2.shape();
    const TensorShape& sigma2_shape = sigma2.shape();
    const TensorShape& mu_shape = mu.shape();
    const TensorShape& lambda_shape = lambda.shape();
    const TensorShape& inveta_shape = inveta.shape();
    const TensorShape& dt_shape = dt.shape();
    const TensorShape& epsilon1_shape = epsilon1.shape();
    const TensorShape& epsilon2_shape = epsilon2.shape();
    const TensorShape& sigma1_shape = sigma1.shape();
    
    
    DCHECK_EQ(grad_sigma2_shape.dims(), 2);
    DCHECK_EQ(sigma2_shape.dims(), 2);
    DCHECK_EQ(mu_shape.dims(), 1);
    DCHECK_EQ(lambda_shape.dims(), 1);
    DCHECK_EQ(inveta_shape.dims(), 1);
    DCHECK_EQ(dt_shape.dims(), 0);
    DCHECK_EQ(epsilon1_shape.dims(), 2);
    DCHECK_EQ(epsilon2_shape.dims(), 2);
    DCHECK_EQ(sigma1_shape.dims(), 2);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_mu_shape(mu_shape);
    TensorShape grad_lambda_shape(lambda_shape);
    TensorShape grad_inveta_shape(inveta_shape);
    TensorShape grad_dt_shape(dt_shape);
    TensorShape grad_epsilon1_shape(epsilon1_shape);
    TensorShape grad_epsilon2_shape(epsilon2_shape);
    TensorShape grad_sigma1_shape(sigma1_shape);
            
    // create output tensor
    
    Tensor* grad_mu = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_mu_shape, &grad_mu));
    Tensor* grad_lambda = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_lambda_shape, &grad_lambda));
    Tensor* grad_inveta = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_inveta_shape, &grad_inveta));
    Tensor* grad_dt = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(3, grad_dt_shape, &grad_dt));
    Tensor* grad_epsilon1 = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(4, grad_epsilon1_shape, &grad_epsilon1));
    Tensor* grad_epsilon2 = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(5, grad_epsilon2_shape, &grad_epsilon2));
    Tensor* grad_sigma1 = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(6, grad_sigma1_shape, &grad_sigma1));
    
    // get the corresponding Eigen tensors for data access
    
    auto mu_tensor = mu.flat<double>().data();
    auto lambda_tensor = lambda.flat<double>().data();
    auto inveta_tensor = inveta.flat<double>().data();
    auto dt_tensor = dt.flat<double>().data();
    auto epsilon1_tensor = epsilon1.flat<double>().data();
    auto epsilon2_tensor = epsilon2.flat<double>().data();
    auto sigma1_tensor = sigma1.flat<double>().data();
    auto grad_sigma2_tensor = grad_sigma2.flat<double>().data();
    auto sigma2_tensor = sigma2.flat<double>().data();
    auto grad_mu_tensor = grad_mu->flat<double>().data();
    auto grad_lambda_tensor = grad_lambda->flat<double>().data();
    auto grad_inveta_tensor = grad_inveta->flat<double>().data();
    auto grad_dt_tensor = grad_dt->flat<double>().data();
    auto grad_epsilon1_tensor = grad_epsilon1->flat<double>().data();
    auto grad_epsilon2_tensor = grad_epsilon2->flat<double>().data();
    auto grad_sigma1_tensor = grad_sigma1->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    int ng = mu_shape.dim_size(0);
    ViscoelasticityStressUpdateBackward(
      grad_epsilon1_tensor, grad_epsilon2_tensor, grad_sigma1_tensor, grad_mu_tensor, grad_inveta_tensor, 
        grad_lambda_tensor, grad_sigma2_tensor, epsilon1_tensor, epsilon2_tensor, sigma1_tensor, 
        mu_tensor, inveta_tensor, lambda_tensor, *dt_tensor, 0, ng);
    
  }
};
REGISTER_KERNEL_BUILDER(Name("ViscoelasticityStressUpdateGrad").Device(DEVICE_CPU), ViscoelasticityStressUpdateGradOp);
