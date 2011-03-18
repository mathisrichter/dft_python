import DynamicField
import GaussKernel

def main():
    interaction_kernel = GaussKernel.GaussKernel(1)
    interaction_kernel.add_mode(1.0, [0.5], [0.0])
    interaction_kernel.add_mode(-5.5, [5.5], [0.0])
    interaction_kernel.calculate()

    field0 = DynamicField.DynamicField([5], interaction_kernel)
    field1 = DynamicField.DynamicField([5], interaction_kernel)

    weight = DynamicField.WeightProcessingStep([-1.0, -0.2, 5.0, -0.2, -1.0])

    DynamicField.connect(field0, weight)
    DynamicField.connect(weight, field1)

    for i in range(0, 500):
        field0.step()
        weight.step()
        field1.step()
        if i == 150:
           field0.set_boost(5) 
        print 'field0 activation: ', field0.get_activation()
        print 'field1 activation: ', field1.get_activation()

if __name__ == "__main__":
    main()
