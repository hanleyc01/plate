(define (null? xs)
  (eq? xs nil))

(define (length xs)
  (if (null? xs)
      0
      (+ 1 (length (cdr xs)))))

(define (map f xs)
  (if (null? xs)
      nil
      (cons (f (car xs))
            (map f (cdr xs)))))

(define (append xs ys)
  (if (null? xs)
      ys
      (cons (car xs)
            (append (cdr xs) ys))))

(define (reverse xs)
  (if (null? xs)
      nil
      (append (reverse (cdr xs))
              (cons (car xs) nil))))

(define nums (cons 1 (cons 2 (cons 3 nil))))

(length (map (lambda (x) (* x x)) nums))
(reverse (append nums nums))
