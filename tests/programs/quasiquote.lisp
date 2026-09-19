(define (makesum a b)
  `(+ ,a ,b))

(define (makeadder n)
  `(lambda (x) (+ x ,n)))

(define (point x y)
  (quasiquote (point (x (unquote x))
                     (y (unquote y)))))

(makesum 1 (* 2 3))
`(total ,(makesum 1 2) is ,(+ 1 2))
`(literal 'sym ,x)
`(outer `(inner ,x))
`,x
