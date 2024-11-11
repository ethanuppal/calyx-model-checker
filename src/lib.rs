#![forbid(unsafe_code)]

use std::{
    any::{type_name, Any},
    collections::{HashMap, VecDeque},
    fmt::{self, Debug},
    hash::{Hash, Hasher},
    marker::PhantomData,
    rc::Rc,
};

use dense_map::{DenseAssociationMap, DenseInternedMap, DenseMap, DenseSet, Id, UniqueId};
use derivative::Derivative;

pub mod dense_map;

#[derive(Debug)]
pub enum Info {
    Op {
        name: String,
        inputs: Vec<String>,
        outputs: Vec<String>,
    },
    Property {
        name: String,
    },
    Spec {
        description: String,
    },
}

impl Info {
    pub fn name_like(&self) -> &str {
        match self {
            Info::Op { name, .. } => name,
            Info::Property { name } => name,
            Info::Spec { description } => description,
        }
    }
}

#[derive(Derivative)]
#[derivative(Default(bound = "T: From<String>"))]
struct NameGen<T: From<String>> {
    names: HashMap<String, usize>,
    generic: PhantomData<T>,
}

impl<T: From<String>> NameGen<T> {
    fn gen<S: AsRef<str>>(&mut self, name: S) -> T {
        let suffix = self.names.entry(name.as_ref().to_string()).or_insert(0);
        let suffixed_name = format!("{}{}", name.as_ref(), suffix);
        *suffix += 1;
        T::from(suffixed_name)
    }
}

#[derive(Derivative)]
#[derivative(Default(bound = "T: Clone"))]
pub struct Context<T: Clone> {
    info_map: DenseMap<Info>,
    base_properties: DenseMap<BaseProperty<T>>,
    properties: DenseInternedMap<Property<T>>,
    specs: DenseMap<Spec<T>>,
    ops: DenseMap<Op<T>>,
    var_gen: NameGen<Var>,
    vars: DenseMap<Var>,
}

struct BaseContext<'a, T: Clone> {
    info_map: &'a DenseMap<Info>,
    base_properties: &'a DenseMap<BaseProperty<T>>,
    properties: &'a DenseInternedMap<Property<T>>,
    specs: &'a DenseMap<Spec<T>>,
    ops: &'a DenseMap<Op<T>>,
}

impl<T: Clone> Context<T> {
    fn base(&self) -> BaseContext<T> {
        BaseContext {
            info_map: &self.info_map,
            base_properties: &self.base_properties,
            properties: &self.properties,
            specs: &self.specs,
            ops: &self.ops,
        }
    }

    fn split(&mut self) -> (BaseContext<T>, &mut DenseMap<Var>) {
        (
            BaseContext {
                info_map: &self.info_map,
                base_properties: &self.base_properties,
                properties: &self.properties,
                specs: &self.specs,
                ops: &self.ops,
            },
            &mut self.vars,
        )
    }
}

pub struct Var {
    name: String,
    backend: Option<Box<dyn FnMut() -> Box<dyn Any>>>,
}

impl From<String> for Var {
    fn from(value: String) -> Self {
        Self {
            name: value,
            backend: None,
        }
    }
}

impl Var {
    fn via_strategy<S: Strategy + 'static>(self) -> Self
    where
        S::Yield: 'static,
    {
        let mut strategy = S::default();
        Self {
            name: self.name,
            backend: Some(Box::new(move || Box::new(strategy.next()))),
        }
    }
}

impl PartialEq for Var {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }
}

impl Eq for Var {}

impl Hash for Var {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.name.hash(state)
    }
}

impl Debug for Var {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Var({})", self.name)
    }
}

pub struct EvaluationContext<'vars> {
    vars: &'vars mut DenseMap<Var>,
    // currently boxes everything which is inefficient
    store: &'vars mut HashMap<Id<Var>, Box<dyn Any>>,
}

impl<'vars> EvaluationContext<'vars> {
    fn init(&mut self, var: Id<Var>) {
        if !self.store.contains_key(&var) {
            if let Some(backend) = &mut self.vars[var].backend {
                self.store.insert(var, (backend)());
            }
        }
    }

    fn insert<T: Any>(&mut self, var: Id<Var>, value: T) {
        if self.vars[var].backend.is_none() {
            self.store.insert(var, Box::new(value));
        }
    }

    fn var(&self, var: Id<Var>) -> &Var {
        &self.vars[var]
    }

    fn get<T: Any>(&self, var: Id<Var>) -> &T {
        self.store
            .get(&var)
            .unwrap_or_else(|| {
                panic!(
                    "{:?} does not exist in the current evaluation context",
                    self.vars[var]
                )
            })
            .downcast_ref()
            .unwrap_or_else(|| panic!("{:?} was not of type {}", self.vars[var], type_name::<T>()))
    }
}

pub trait Strategy: Default {
    type Yield;

    fn next(&mut self) -> Self::Yield;
}

#[derive(Derivative)]
#[derivative(Default)]
pub struct Arbitrary<T> {
    rng: rand::rngs::ThreadRng,
    generic: PhantomData<T>,
}

impl<T> Strategy for Arbitrary<T>
where
    rand::distributions::Standard: rand::distributions::Distribution<T>,
{
    type Yield = T;

    fn next(&mut self) -> Self::Yield {
        use rand::Rng;

        self.rng.gen()
    }
}

pub struct Op<T: Clone> {
    info: Id<Info>,
    outputs: Vec<Id<Var>>,
    f: Box<dyn Fn(&mut T, &mut EvaluationContext, &[Id<Var>]) -> Vec<Id<Var>>>,
}

pub struct BaseProperty<T: Clone> {
    info: Id<Info>,
    f: Rc<dyn Fn(&T) -> bool>,
}

#[derive(Derivative)]
#[derivative(PartialEq(bound = "T: Clone"))]
#[derivative(Eq(bound = "T: Clone"))]
#[derivative(Hash(bound = "T: Clone"))]
#[derivative(Clone(bound = "T: Clone"))]
pub enum Property<T: Clone> {
    Base(Id<BaseProperty<T>>),
    Not { child: UniqueId<Property<T>> },
}

impl<T: Clone> Property<T> {
    fn evaluate(&self, state: &T, context: &BaseContext<T>) -> bool {
        match self {
            Property::Base(base) => (context.base_properties[*base].f)(state),
            Property::Not { child } => !context.properties[*child].evaluate(state, context),
        }
    }

    fn to_string(&self, context: &BaseContext<T>) -> String {
        match self {
            Property::Base(base) => {
                let info = context.base_properties[*base].info;
                context.info_map[info].name_like().to_string()
            }
            Property::Not { child } => {
                let child = &context.properties[*child];
                format!("~({})", child.to_string(context))
            }
        }
    }
}

enum Step<T: Clone> {
    Op(Id<Op<T>>, Vec<Id<Var>>),
    Require(UniqueId<Property<T>>),
    AssertEq(Id<Var>, Id<Var>, Box<dyn Fn(&EvaluationContext) -> bool>),
}

#[derive(Derivative)]
#[derivative(Default(bound = "T: Clone"))]
pub struct Workload<T: Clone> {
    steps: Vec<Step<T>>,
}

impl<T: Clone> Workload<T> {
    fn enqueue(&mut self, step: Step<T>) {
        self.steps.push(step);
    }
}

pub struct Spec<T: Clone> {
    info: Id<Info>,
    inputs: Vec<Id<Var>>,
    preconditions: Vec<UniqueId<Property<T>>>,
    workload: Workload<T>,
}

pub struct SpecBuilder<'context, T: Clone> {
    context: &'context mut Context<T>,
    spec: Spec<T>,
}

impl<'context, T: Clone> SpecBuilder<'context, T> {
    pub fn when(&mut self, preconditions: impl IntoIterator<Item = UniqueId<Property<T>>>) {
        self.spec.preconditions.extend(preconditions);
    }

    pub fn var<S: Strategy + 'static>(&mut self, name: &str) -> Id<Var> {
        let var = self.context.new_input_var::<S, _>(name);
        self.spec.inputs.push(var);
        var
    }

    pub fn run(&mut self, op: Id<Op<T>>, arguments: impl IntoIterator<Item = Id<Var>>) {
        self.spec
            .workload
            .enqueue(Step::Op(op, arguments.into_iter().collect()));
    }

    pub fn call<const N: usize>(
        &mut self,
        op: Id<Op<T>>,
        arguments: impl IntoIterator<Item = Id<Var>>,
    ) -> [Id<Var>; N] {
        self.spec
            .workload
            .enqueue(Step::Op(op, arguments.into_iter().collect()));
        let op = &self.context.ops[op];
        let outputs = &op.outputs;
        if outputs.len() != N {
            panic!(
                "{} produces {} output(s), but user call requested {}",
                self.context.info_map[op.info].name_like(),
                outputs.len(),
                N
            );
        }
        let mut result = [Id::INVALID; N];
        result.copy_from_slice(outputs);
        result
    }

    pub fn require(&mut self, condition: UniqueId<Property<T>>) {
        self.spec.workload.enqueue(Step::Require(condition))
    }

    pub fn assert_eq<U: PartialEq + 'static>(&mut self, lhs: Id<Var>, rhs: Id<Var>) {
        self.spec.workload.enqueue(Step::AssertEq(
            lhs,
            rhs,
            Box::new(move |evaluation_context| {
                evaluation_context.get::<U>(lhs) == evaluation_context.get::<U>(rhs)
            }),
        ));
    }
}

impl<T: Clone> Context<T> {
    pub fn property<S: AsRef<str>, F: Fn(&T) -> bool + 'static>(
        &mut self,
        name: S,
        condition: F,
    ) -> UniqueId<Property<T>> {
        let info = self.info_map.add(Info::Property {
            name: name.as_ref().to_string(),
        });
        let base_property = self.base_properties.add(BaseProperty {
            info,
            f: Rc::new(condition),
        });
        self.properties.insert(Property::Base(base_property))
    }

    pub fn negated(&mut self, property: UniqueId<Property<T>>) -> UniqueId<Property<T>> {
        self.properties.insert(Property::Not { child: property })
    }

    pub fn op_1arg<S: AsRef<str>, F: Fn(&mut T, Arg1) + 'static, Arg1: Clone + 'static>(
        &mut self,
        name: S,
        f: F,
    ) -> Id<Op<T>> {
        let info = self.info_map.add(Info::Op {
            name: name.as_ref().to_string(),
            inputs: vec![type_name::<Arg1>().to_string()],
            outputs: vec![],
        });
        self.ops.add(Op {
            info,
            outputs: vec![],
            f: Box::new(move |value, context, inputs| {
                assert_eq!(1, inputs.len());
                let arg1 = context.get::<Arg1>(inputs[0]).clone();
                f(value, arg1);
                vec![]
            }),
        })
    }

    pub fn op_ret<S: AsRef<str>, F: Fn(&mut T) -> Result + 'static, Result: Clone + 'static>(
        &mut self,
        name: S,
        f: F,
    ) -> Id<Op<T>> {
        let info = self.info_map.add(Info::Op {
            name: name.as_ref().to_string(),
            inputs: vec![],
            outputs: vec![type_name::<Result>().to_string()],
        });
        let var = self.new_intermediate_var(format!("{}_result", name.as_ref()));
        self.ops.add(Op {
            info,
            outputs: vec![var],
            f: Box::new(move |value, context, inputs| {
                assert_eq!(0, inputs.len());
                let result = f(value);
                context.insert(var, result.clone());
                vec![var]
            }),
        })
    }

    pub fn spec<S: AsRef<str>, F: FnMut(&mut SpecBuilder<T>)>(
        &mut self,
        description: S,
        mut f: F,
    ) -> Id<Spec<T>> {
        let info = self.info_map.add(Info::Spec {
            description: description.as_ref().to_string(),
        });
        let mut builder = SpecBuilder {
            context: self,
            spec: Spec {
                info,
                inputs: Vec::default(),
                preconditions: Vec::default(),
                workload: Workload::default(),
            },
        };
        f(&mut builder);
        let spec = builder.spec;
        self.specs.add(spec)
    }

    fn new_intermediate_var<S: AsRef<str>>(&mut self, name: S) -> Id<Var> {
        let var = self.var_gen.gen(name);
        self.vars.add(var)
    }

    fn new_input_var<S: Strategy + 'static, S2: AsRef<str>>(&mut self, name: S2) -> Id<Var> {
        let var = self.var_gen.gen(name).via_strategy::<S>();
        self.vars.add(var)
    }
}

type PropertySet<T> = DenseSet<UniqueId<Property<T>>>;

struct SearchNode<T: Clone> {
    state: T,
    properties: PropertySet<T>,
}

impl<T: Clone> SearchNode<T> {
    pub fn from(value: T, context: &Context<T>) -> Self {
        let mut properties = DenseSet::default();
        for property in context.properties.keys() {
            if context.properties[property].evaluate(&value, &context.base()) {
                properties.insert(property);
            }
        }
        Self {
            state: value,
            properties,
        }
    }
}

pub struct Engine<'context, T: Clone> {
    initial: T,
    context: &'context mut Context<T>,
    bfs: VecDeque<SearchNode<T>>,
    specs: DenseAssociationMap<Spec<T>, PropertySet<T>>,
}

impl<'context, T: Clone> Engine<'context, T> {
    pub fn new(initial: T, context: &'context mut Context<T>) -> Self {
        Self {
            initial,
            context,
            bfs: VecDeque::default(),
            specs: DenseAssociationMap::default(),
        }
    }

    pub fn clear_specs(&mut self) {
        self.bfs.clear();
        self.specs.clear();
    }

    pub fn add_spec(&mut self, spec: Id<Spec<T>>) {
        let mut property_set = DenseSet::default();
        for precondition in &self.context.specs[spec].preconditions {
            property_set.insert(*precondition);
        }
        self.specs.associate(spec, property_set);
    }

    pub fn search(&mut self, max_iters: usize) {
        self.bfs.clear();
        self.bfs
            .push_back(SearchNode::from(self.initial.clone(), self.context));
        let mut iters = 0;
        let mut var_store = HashMap::default();

        while iters < max_iters && !self.bfs.is_empty() {
            let next = self.bfs.pop_front().unwrap();
            for spec in self.find_applicable_specs(next.properties) {
                let mut to_add = next.state.clone();
                self.apply_spec(&mut to_add, &mut var_store, spec);
                self.bfs.push_back(SearchNode::from(to_add, self.context));
            }

            iters += 1;
        }
    }

    fn find_applicable_specs(&self, properties: PropertySet<T>) -> Vec<Id<Spec<T>>> {
        let mut applicable = vec![];
        for (spec, preconditions) in self.specs.iter() {
            if preconditions.is_subset_of(&properties) {
                applicable.push(spec);
            } else {
                let info = self.context.specs[spec].info;
                let info = &self.context.info_map[info];
                panic!(
                    "spec {} could not be applied: {:?} not a subset of {:?}",
                    info.name_like(),
                    preconditions,
                    properties
                );
            }
        }
        applicable
    }

    fn apply_spec(
        &mut self,
        value: &mut T,
        var_store: &mut HashMap<Id<Var>, Box<dyn Any>>,
        spec: Id<Spec<T>>,
    ) {
        let (base_context, vars) = self.context.split();
        let spec = &base_context.specs[spec];

        for input in &spec.inputs {
            EvaluationContext {
                vars,
                store: var_store,
            }
            .init(*input);
        }

        for step in &spec.workload.steps {
            match step {
                Step::Op(op, inputs) => {
                    let mut evaluation_context = EvaluationContext {
                        vars,
                        store: var_store,
                    };
                    (base_context.ops[*op].f)(value, &mut evaluation_context, inputs);
                }
                Step::Require(property) => {
                    let property = &base_context.properties[*property];
                    if !property.evaluate(value, &base_context) {
                        panic!("property `{}` violated", property.to_string(&base_context));
                    }
                }
                Step::AssertEq(lhs, rhs, check) => {
                    let evaluation_context = EvaluationContext {
                        vars,
                        store: var_store,
                    };
                    if !check(&evaluation_context) {
                        panic!(
                            "assertion failed: {:?} != {:?}",
                            evaluation_context.var(*lhs),
                            evaluation_context.var(*rhs)
                        )
                    }
                }
            }
        }
    }
}
