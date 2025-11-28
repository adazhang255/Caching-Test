.data
 
 # array terminated by 0 (which is not part of the array)
 xarr:
 .word 2, 4, 6, 8, 10, 0
 .data
 
 arrow: .asciiz " -> "
 
 .text
 
 main:
     li      $sp,        0x7ffffffc      # initialize $sp
 
 # PROLOGUE
     subu    $sp,        $sp,        8   # expand stack by 8 bytes
     sw      $ra,        8($sp)          # push $ra (ret addr, 4 bytes)
     sw      $fp,        4($sp)          # push $fp (4 bytes)
     addu    $fp,        $sp,        8   # set $fp to saved $ra
 
     subu    $sp,        $sp,        12  # save s0 and s1 on stack before usi    ng them
     sw      $s0,        12($sp)         # push $s0
     sw      $s1,        8($sp)          # push $s1
     sw      $s2,        4($sp)          # push $s2
 
     la      $s0,        xarr            # load address to s0
 
 main_for:
     lw      $s1,        ($s0)           # use s1 for xarr[i] value
     li      $s2,        0               # use s2 for initial depth (steps)
     beqz    $s1,        main_end        # if xarr[i] == 0, stop.
 
 # save args on stack rightmost one first
     subu    $sp,        $sp,        8   # save args on stack
     sw      $s2,        8($sp)          # save depth
     sw      $s1,        4($sp)          # save xarr[i]

	 li      $v0,        1
	 move    $a0,        $s1             # print_int(xarr[i])
 	syscall
 
	li      $v0,        4               # print " -> "
     la      $a0,        arrow
     syscall
 
 jal     collatz                     # result = collatz(xarr[i])
     move    $a0,        $v0             # print_int(result)
     li      $v0,        1
     syscall
 
     li      $a0,        10              # print_char('\n')
     li      $v0,        11
     syscall
 
     addu    $s0,        $s0,        4   # make s0 point to the next element
 
     lw      $s2,        8($sp)          # restore depth
     lw      $s1,        4($sp)          # restore xarr[i]
     addu    $sp,        $sp,        8   # shrink stack
     j       main_for

main_end:
     lw      $s0,        12($sp)         # restore $s0
     lw      $s1,        8($sp)          # restore $s1
     lw      $s2,        4($sp)          # restore $s2
 
 # EPILOGUE
     move    $sp,        $fp             # restore $sp
     lw      $ra,        ($fp)           # restore saved $ra
     lw      $fp,        -4($sp)         # restore saved $fp
     jr      $ra                         # return to kernel
 
 collatz:
     # PROLOGUE
     subu    $sp, $sp, 8         # make space for pointers
     sw      $ra, 8($sp)         # save (push) $ra onto the stack
     sw      $fp, 4($sp)         # save $fp
     addu    $fp, $sp, 8         # set $fp to the saved $ra
 
     # BODY
     subu    $sp, $sp, 8         # make space for locals
     sw      $s0, 8($sp)         # callee save $s0
     sw      $s1, 4($sp)         # callee save $s1
     lw   $s0, 4($fp)  # store n from stack
	lw   $s1, 8($fp)  # store d from stack

	if:
	# if (n != 1) 
	li      $t0, 1              # store 1 constant for comparison
	beq     $s0, $t0, base_case # if equal, go to base_case
#	 if (n % 2)
	rem     $t0, $s0, 2         # compute n % 2 
	beq     $t0, $0, else       # if n is even go to else  

	# otherwise, evaluate as odd 
	# return collatz(3 * n + 1, d + 1);
	li      $t0, 3              # store 3
	mul     $t1, $s0, $t0       # store 3 * n
	addu    $a0, $t1, 1         # pass 3 * n + 1 as the first arg
	addu    $a1, $s1, 1         # pass d + 1 as the second arg 
	# caller-saved registers before function call
	subu    $sp, $sp, 16        # make space for $t0, $t1, $a0, $a1,
	sw      $t0, 16($sp)        # caller save $t0
	sw      $t1, 12($sp)        # caller save $t1
	sw      $a0, 4($sp)         # caller save $a0
	sw      $a1, 8($sp)         # caller save $a1
	jal     collatz             # recursively call
	
	# restore caller saved registers 
	lw      $t0, 16($sp)        # restore caller $t0
	lw      $t1, 12($sp)        # restore caller $t1
	lw      $a0, 4($sp)         # restore caller $a0
	lw      $a1, 8($sp)         # restore caller $a1
	addu    $sp, $sp, 16        # clean up stack
    b       end_recur           # exit

	else:
	# return collatz(n / 2, d + 1);
	# caller-saved registers before function call (for redundancy)
	srl     $a0, $s0, 1         # store n/2 as first arg
	addu	$a1, $s1, 1         # pass d + 1 as the second arg 
	subu    $sp, $sp, 16        # make space for $t0, $t1, $a0, $a1,
	sw      $t0, 16($sp)        # caller save $t0
	sw      $t1, 12($sp)        # caller save $t1
	sw      $a0, 4($sp)         # caller save $a0
	sw      $a1, 8($sp)         # caller save $a1
	jal     collatz             # recursively call
	# restore caller saved registers 
	lw      $t0, 16($sp)        # restore caller $t0
	lw      $t1, 12($sp)        # restore caller $t1
	lw      $a0, 4($sp)         # restore caller $a0
	lw      $a1, 8($sp)         # restore caller $a1
	addu    $sp, $sp, 16        # clean up stack
    b       end_recur           # exit

	base_case:
	move    $v0, $s1            # return d;

    end_recur:
	# restore callee saved registers
	lw      $s0, 8($sp)        # restore callee $s0
	lw      $s1, 4($sp)        # restore callee $s1

	# EPILOGUE   
	move    $sp, $fp            # restore $sp to $fp (callee stack frame)
	lw      $ra, ($fp)          # restore $ra (at "top" of fp stack)
	lw      $fp, -4($sp)        # restore original $fp (caller stack frame)
	# return to caller 
	jr	$ra
