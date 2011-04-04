import DynamicField
import Kernel
import numpy

def main():
    interaction_kernel_0 = Kernel.GaussKernel(3)
    interaction_kernel_0.add_mode(1.0, [0.5,0.5,0.5], [0.0,0.0,0.0])
    interaction_kernel_0.add_mode(-5.5, [5.5,5.5,5.5], [0.0,0.0,0.0])
    interaction_kernel_0.calculate()

    interaction_kernel_1 = Kernel.GaussKernel(1)
    interaction_kernel_1.add_mode(1.0, [0.5], [0.0])
    interaction_kernel_1.add_mode(-5.5, [5.5], [0.0])
    interaction_kernel_1.calculate()

    input_field = DynamicField.DynamicField([[5],[2]], [], None)
    input_field.set_boost(20)
    blub = numpy.array([[0.0,0.5,1.0,0.5,0.0],[0.0,0.25,0.5,0.25,0.0]])
    blub = blub.transpose() * 40 
    input_weight = DynamicField.Weight(blub)

    field_0 = DynamicField.DynamicField([[5],[4],[2]], [], None)
    field_1 = DynamicField.DynamicField([[4],[6],[10]], [], None)


    scaler = DynamicField.Scaler()

#    weight = DynamicField.Weight([0., 0.5, 1.0, 0.5, 0.0])
#    weight = DynamicField.Weight(5.)

    projection_0 = DynamicField.Projection(2, 2, set([0,1]), [1,0])

    processing_steps = [scaler]
    
    DynamicField.connect(field_0, field_1, processing_steps)
    #DynamicField.connect(input_field, field_0, [input_weight])


    for i in range(0, 500):
       # input_field.step()
       # input_weight.step()
        field_0.step()
        print("field 0:")
        print(field_0.get_output())
        print(field_0.get_output().shape)
        for processing_step in processing_steps:
            processing_step.step()
            print(processing_step.get_name() + ": ")
            print(processing_step.get_output())
            print(processing_step.get_output().shape)
        field_1.step()
        print("field 1:")
        print(field_1.get_activation())
        print(field_1.get_activation().shape)

if __name__ == "__main__":
    main()
