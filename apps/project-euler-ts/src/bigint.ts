/**
 * A big integer represented as an array of digits (least significant first).
 */
export class BigInt {
  private digits: number[];

  private constructor(digits: number[]) {
    this.digits = digits;
  }

  /** Create a BigInt from a decimal string. */
  static fromString(s: string): BigInt {
    const digits = s
      .split("")
      .reverse()
      .map((c) => parseInt(c, 10))
      .filter((d) => !isNaN(d));
    return new BigInt(digits);
  }

  /** Create a BigInt representing zero. */
  static zero(): BigInt {
    return new BigInt([0]);
  }

  /** Add two BigInts using elementary school addition with carry. */
  add(other: BigInt): BigInt {
    const result: number[] = [];
    let carry = 0;
    const maxLen = Math.max(this.digits.length, other.digits.length);

    for (let i = 0; i < maxLen; i++) {
      const a = this.digits[i] ?? 0;
      const b = other.digits[i] ?? 0;
      const sum = a + b + carry;
      result.push(sum % 10);
      carry = Math.floor(sum / 10);
    }

    if (carry > 0) {
      result.push(carry);
    }

    return new BigInt(result);
  }

  /** Convert to decimal string. */
  toString(): string {
    if (this.digits.length === 0) {
      return "0";
    }
    return this.digits.slice().reverse().join("");
  }
}
