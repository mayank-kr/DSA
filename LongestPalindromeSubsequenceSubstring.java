public class LongestPalindromeSubsequenceSubstring {
    public static void main(String[] args){
        System.out.println(longestPalindrome("TMENENT"));
    }
    public static String longestPalindrome(String s) {
        String ans = "";
        for (int i = 0; i<s.length() ; i++){
            if (check(s,i).length() > ans.length()){
                ans = check(s,i);
            }
        }
        return s;
    }
    public static String check(String s, int index){
        int a = index;
        int b = index;
        String s1 = "";
        String s2 = "";
        while (a>=0 && b<s.length() && s.charAt(a)==s.charAt(b)){
            a--;
            b++;
        }
        s1 = s.substring(a+1,b);
        a = index;
        b = index+1;
        while (a>=0 && b<s.length() && s.charAt(a)==s.charAt(b)){
            a--;
            b++;
        }
        s2 = s.substring(a+1,b);
        return (s1.length()>=s2.length())?s1:s2;
    }
}
