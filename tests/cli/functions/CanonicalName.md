# `CanonicalName`

Gives the standard name of an entity, entity class or entity property.

```scrut
$ wo 'CanonicalName[Entity["Star", "Sirius"]]'
Sirius
```

Lists of entities are mapped over:

```scrut
$ wo 'CanonicalName[{Entity["Planet", "Earth"], EntityProperty["Country", "Population"]}]'
{Earth, Population}
```
