use super::linked_list::LinkedList;

pub struct Queue<T> {
    list: LinkedList<T>,
}

impl<T> Queue<T> {
    pub fn new() -> Self {
        Queue { list: LinkedList::new() }
    }

    pub fn enqueue(&mut self, elem: T) {
        self.list.push_back(elem);
    }

    pub fn dequeue(&mut self) -> Option<T> {
        self.list.pop_front()
    }

    pub fn peek(&self) -> Option<&T> {
        self.list.peek()
    }

    pub fn peek_mut(&mut self) -> Option<&mut T> {
        self.list.peek_mut()
    }
}

#[cfg(test)]
mod test {
    use super::Queue;

    #[test]
    fn queue_basics() {
        let mut queue = Queue::new();

        // Check empty queue behaves right
        assert_eq!(queue.dequeue(), None);

        // Enqueue items
        queue.enqueue(1);
        queue.enqueue(2);
        queue.enqueue(3);

        // Check normal removal (FIFO)
        assert_eq!(queue.dequeue(), Some(1));
        assert_eq!(queue.dequeue(), Some(2));

        // Enqueue some more
        queue.enqueue(4);
        queue.enqueue(5);

        // Check normal removal
        assert_eq!(queue.dequeue(), Some(3));
        assert_eq!(queue.dequeue(), Some(4));

        // Check exhaustion
        assert_eq!(queue.dequeue(), Some(5));
        assert_eq!(queue.dequeue(), None);
    }
    
    #[test]
    fn queue_peek() {
        let mut queue = Queue::new();
        assert_eq!(queue.peek(), None);
        assert_eq!(queue.peek_mut(), None);

        queue.enqueue(1);
        queue.enqueue(2);
        
        assert_eq!(queue.peek(), Some(&1));
        
        queue.peek_mut().map(|val| *val = 42);
        assert_eq!(queue.peek(), Some(&42));
        
        assert_eq!(queue.dequeue(), Some(42));
    }
}
