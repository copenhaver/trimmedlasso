% test file

cvx_begin
    variable x binary
    variable y
    minimize( x + y  )
    subject to
        x >= 0;
        y >= 0;
        x + y >= 1;
        x*y == 0;
cvx_end