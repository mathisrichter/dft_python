import BehavioralOrganization as BehOrg
import DynamicField
import numpy

def main():
    task_node = DynamicField.DynamicField([], [], None)

    int_weight = numpy.array([[1.0, 1.0, 1.0],
                              [1.2, 2.0, 1.2],
                              [2.0, 6.0, 2.0],
                              [1.2, 2.0, 1.2],
                              [1.0, 1.0, 1.0]])

    elem_behavior_0 = BehOrg.ElementaryBehavior(field_dimensionality=2,
                                                field_sizes=[[5],[3]],
                                                field_resolutions=[],
                                                int_node_to_int_field_weight=int_weight,
                                                cos_field_to_cos_node_weight=6.0,
                                                cos_node_to_cos_memory_node_weight=6.0,
                                                int_inhibition_weight=-6.0,
                                                reactivating=False)

    elem_behavior_1 = BehOrg.ElementaryBehavior(field_dimensionality=2,
                                                field_sizes=[[5],[3]],
                                                field_resolutions=[],
                                                int_node_to_int_field_weight=5.5,
                                                cos_field_to_cos_node_weight=5.5,
                                                cos_node_to_cos_memory_node_weight=5.5,
                                                int_inhibition_weight=-5.5,
                                                reactivating=False)

    BehOrg.connect_to_task(task_node, elem_behavior_0)
    BehOrg.connect_to_task(task_node, elem_behavior_1)

    BehOrg.precondition(elem_behavior_0, elem_behavior_1, task_node)


    for i in range(2000):

        if (i > 20):
            task_node.set_boost(20)
        if (i > 200):
            elem_behavior_0.get_cos_field().set_boost(5.5)
        if (i > 400):
            elem_behavior_0.get_cos_field().set_boost(0.0)
        task_node.step()
        elem_behavior_0.step()
        elem_behavior_1.step()

        print("task node activation (boost: " + str(task_node.get_boost()) + ")")
        print(task_node.get_activation())

        print("int node activation")
        print(elem_behavior_0.get_intention_node().get_activation())
        print("int field activation")
        print(elem_behavior_0.get_intention_field().get_activation())
        print("cos field activation (boost: " + str(elem_behavior_0.get_cos_field().get_boost()) + ")")
        print(elem_behavior_0.get_cos_field().get_activation())
        print("cos node activation")
        print(elem_behavior_0.get_cos_node().get_activation())
        print("cos mem node activation")
        print(elem_behavior_0.get_cos_memory_node().get_activation())

        print("int node activation 1")
        print(elem_behavior_1.get_intention_node().get_activation())
        print("int field activation 1")
        print(elem_behavior_1.get_intention_field().get_activation())
        print("cos field activation 1 (boost: " + str(elem_behavior_1.get_cos_field().get_boost()) + ")")
        print(elem_behavior_1.get_cos_field().get_activation())
        print("cos node activation 1")
        print(elem_behavior_1.get_cos_node().get_activation())
        print("cos mem node activation 1")
        print(elem_behavior_1.get_cos_memory_node().get_activation())


        print("\n")


if __name__ == "__main__":
    main()
