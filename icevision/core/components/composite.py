__all__ = ["Component", "Composite", "TaskComponent", "TaskComposite"]

from icevision.imports import *
from icevision.core import tasks


class Component:
    order = 0.5

    def __init__(self):
        self.composite = None

    def set_composite(self, composite):
        self.composite = composite


class TaskComponent(Component):
    def __init__(self, task=tasks.common):
        self.task = task


class TaskComposite:
    base_components = set()

    def __init__(self, components: Sequence[TaskComponent]):
        components = set(components)
        components.update(set(comp() for comp in self.base_components))
        self.components = components
        self.set_task_components(self.components)

    def __getattr__(self, name):
        if name == "task_composites":
            raise AttributeError(name)

        # TODO: Possible bug if no task with _default is passed
        try:
            return getattr(self.task_composites[tasks.common.name], name)
        except AttributeError:
            pass

        try:
            return self.task_composites[name]
        except KeyError:
            pass

        raise AttributeError(f"{self.__class__.__name__} has no attribute {name}")

    def add_component(self, component: TaskComponent):
        self.components.add(component)
        self.set_task_components(self.components)

    def remove_component_by_type(self, component_type: TaskComponent):
        for component in self.components:
            if isinstance(component, component_type):
                break
        self.components.remove(component)
        self.set_task_components(self.components)

    def set_task_components(self, components: Sequence[TaskComponent]):
        task_components = defaultdict(list)
        # example: task_components['detect'] = (LabelsComponent, BBoxesComponent, ...)
        for component in components:
            task_components[component.task].append(component)

        self.task_composites = OrderedDict()
        for task, components in sorted(
            task_components.items(), key=lambda o: o[0].order
        ):
            self.task_composites[task.name] = composite = Composite()
            composite.add_components(components)
            if task != tasks.common:
                composite.set_parent(self)

    # TODO: rename reduce_on_all_tasks_components
    def reduce_on_components(
        self,
        fn_name: str,
        reduction: Optional[str] = None,
        **fn_kwargs,
    ) -> Dict[str, Any]:
        results = {}
        for task, composite in self.task_composites.items():
            result = composite.reduce_on_components(fn_name, reduction, **fn_kwargs)
            results[task] = result

        return results

    def reduce_on_task_components(
        self,
        fn_name: str,
        task_name: str,
        reduction: Optional[str] = None,
        **fn_kwargs,
    ) -> Any:
        composite = self.task_composites[task_name]
        return composite.reduce_on_components(fn_name, reduction, **fn_kwargs)


class Composite:
    base_components = set()

    def __init__(self, components: Optional[Sequence[Component]] = None):
        self._parent = None

        components = set(components) if components is not None else set()
        components.update(set(comp() for comp in self.base_components))
        self.set_components(components)

    def __getattr__(self, name):
        # avoid recursion https://nedbatchelder.com/blog/201010/surprising_getattr_recursion.html
        if name in ["components", "_parent"]:
            raise AttributeError(name)
        # delegates attributes to components
        for component in self.components:
            try:
                return getattr(component, name)
            except AttributeError:
                pass
        # delegates attributes to parent
        try:
            return getattr(self._parent, name)
        except AttributeError:
            pass

        raise AttributeError(f"{self.__class__.__name__} has no attribute {name}")

    def reduce_on_components(
        self, fn_name: str, reduction: Optional[str] = None, **fn_kwargs
    ) -> Any:
        results = []
        for component in self.components:
            results.append(getattr(component, fn_name)(**fn_kwargs))

        if reduction is not None and len(results) > 0:
            out = results.pop(0)
            for r in results:
                getattr(out, reduction)(r)
        else:
            out = results

        return out

    def get_component_by_type(self, component_type) -> Union[Component, None]:
        for component in self.components:
            if isinstance(component, component_type):
                return component

    def add_component(self, component):
        self.add_components([component])

    def add_components(self, components):
        self.set_components(set(components).union(set(self.components)))

    def set_components(self, components):
        self.components = sorted(components, key=lambda o: o.order)
        self.components_cls = [comp.__class__ for comp in self.components]

        for component in self.components:
            component.set_composite(self)

    def set_parent(self, parent):
        self._parent = parent
