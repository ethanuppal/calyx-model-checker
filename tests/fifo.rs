use calyx_ffi::cider_ffi_backend;
use calyx_ffi::prelude::*;

enum QueueCommand {
    Pop = 0,
    Push = 1,
}

#[derive(PartialEq, Eq, Debug)]
enum QueueStatus {
    Ok = 0,
    Err = 1,
}

calyx_ffi::declare_interface! {
    Queue(cmd: 1, value: 32) -> (ans: 32, err: 1, length: 5) impl {
        fn status(&self) -> QueueStatus {
            if self.err() == 0 { QueueStatus::Ok } else { QueueStatus::Err }
        }
    } mut impl {
        fn assert_no_error(&mut self) {
            // assert_eq!(QueueStatus::Ok, self.status(), "queue underflowed or overflowed");
        }

        fn push(&mut self, value: u32) {
            self.set_cmd(QueueCommand::Push as u64);
            self.set_value(value as u64);
            self.go();
            self.assert_no_error();
        }

        fn pop(&mut self) -> u32 {
            self.set_cmd(QueueCommand::Pop as u64);
            self.go();
            self.assert_no_error();
            self.ans() as u32
        }
    }
}

#[calyx_ffi(
    src = "tests/fifo.futil",
    comp = "main",
    backend = cider_ffi_backend,
    derive = [
        Queue(cmd: 1, value: 32) -> (ans: 32, err: 1, length: 5)
    ]
)]
struct Fifo;

#[cfg(test)]
#[calyx_ffi_tests]
mod tests {
    use calyx_model_checker::{Arbitrary, Context, Engine};

    use super::*;

    #[calyx_ffi_test]
    fn test_fifo(fifo: &mut Fifo) {
        println!("testing fifo");

        let mut ctx = Context::<Fifo>::default();

        let no_errors = ctx.property("no errors", |fifo| fifo.status() != QueueStatus::Err);
        let has_errors = ctx.negated(no_errors);

        let is_empty = ctx.property("is empty", |fifo| fifo.length() == 0);
        let not_empty = ctx.negated(is_empty);

        let push_op = ctx.op_1arg("push", |fifo, value| fifo.push(value));
        let pop_op = ctx.op_ret("pop", |fifo| fifo.pop());

        let push_ok = ctx.spec("push should not cause errors", |b| {
            b.when([no_errors]);

            let value = b.var::<Arbitrary<u32>>("value");
            b.run(push_op, [value]);

            b.require(no_errors);
        });

        let pop_ok = ctx.spec("pop when not empty should not cause errors", |b| {
            b.when([no_errors, not_empty]);

            b.run(pop_op, []);

            b.require(no_errors);
        });

        let push_pop = ctx.spec("a pop should return the pushed value when empty", |b| {
            b.when([no_errors, is_empty]);

            let value = b.var::<Arbitrary<u32>>("value");
            b.run(push_op, [value]);
            let [result] = b.call(pop_op, []);

            b.assert_eq::<u32>(value, result);
            b.require(no_errors);
        });

        let bad = ctx.spec("huh", |b| {
            b.when([is_empty]);
            b.run(pop_op, []);
            b.require(no_errors);
        });

        let find_error = ctx.spec("pop empty queue gives error", |b| {
            b.when([is_empty]);
            b.run(pop_op, []);
            b.require(has_errors);
        });

        let errors_persist = ctx.spec("errors persist", |b| {
            b.when([has_errors]);

            let value = b.var::<Arbitrary<u32>>("value");
            b.run(push_op, [value]);

            b.require(has_errors);
        });

        let mut engine = Engine::new(fifo.clone(), &mut ctx);
        engine.add_spec(push_ok);
        engine.add_spec(pop_ok);
        engine.add_spec(push_pop);
        engine.add_spec(find_error);
        engine.add_spec(errors_persist);
        // engine.add_spec(bad);
        engine.search(10);
    }
}
