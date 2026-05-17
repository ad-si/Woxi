Needs["DatabaseLink`"];
connectDb[dbUser_, dbPass_, dbUrl_] :=
  OpenSQLConnection[JDBC["mysql", dbUrl], "Username" -> dbUser,
   "Password" -> dbPass];
createUser::nameTaken = "The username '`1`' is already taken.";
createUser[dbUser_, dbPass_, dbUrl_, user_, pass_] :=
  Module[{db = connectDb[dbUser, dbPass, dbUrl],
    salt = RandomChoice[Range[32, 127], 16]},
   If[MemberQ[SQLSelect[db, "users", {"username"}], {user}],
    Message[createUser::nameTaken, user]; Return[]];
   SQLInsert[db,
    "users", {"username", "pass_salt", "pass_md5"}, {user,
     SQLBinary[salt],
     SQLBinary[
      IntegerDigits[Hash[FromCharacterCode[salt] <> pass, "MD5"], 256,
        16]]}]; CloseSQLConnection[db];];
authenticateUser[dbUser_, dbPass_, dbUrl_, user_, pass_] :=
  Module[{db = connectDb[dbUser, dbPass, dbUrl], rtn},
   rtn = MemberQ[SQLSelect[db, "users", {"username"}], {user}] &&
     Module[{data =
        SQLSelect[db, "users", {"username", "pass_salt", "pass_md5"},
          SQLColumn["username"] == user][[1]]},
      Hash[FromCharacterCode[data[[2, 1]]] <> pass, "MD5"] ==
       FromDigits[data[[3, 1]], 256]]; CloseSQLConnection[db]; rtn];
