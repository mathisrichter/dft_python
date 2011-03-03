import DynamicField
import GaussKernel

def main():
    interaction_kernel = GaussKernel.GaussKernel(1)
    interaction_kernel.add_mode(1.0, [0.5], [0.0])
    interaction_kernel.add_mode(-5.5, [5.5], [0.0])
    interaction_kernel.calculate()

    field0 = DynamicField.DynamicField([5], interaction_kernel)
    field1 = DynamicField.DynamicField([5], interaction_kernel)

    weight0 = DynamicField.WeightProcessingStep([-1.0, -0.2, 5.0, -0.2, -1.0])
    weight1 = DynamicField.WeightProcessingStep([-1.0, -0.2, 5.0, -0.2, -1.0])

    weight_group = DynamicField.ProcessingGroup()
    weight_group.add_processing_step(weight0)
    weight_group.add_processing_step(weight1)

    DynamicField.connect(field0, weight_group)
    DynamicField.connect(weight_group, field1)
    weight_group.connect_group()

    for i in range(0, 500):
        field0.step()
        weight_group.step()
        field1.step()
        if i == 150:
           field0.set_boost(5) 
        print 'field0 activation: ', field0.get_activation()
        print 'field1 activation: ', field1.get_activation()

if __name__ == "__main__":
    main()
