use proc_macro::TokenStream;
use quote::quote;
use syn::{LitInt, parse_macro_input};

#[proc_macro]
pub fn generate_mlp(input: TokenStream) -> TokenStream {
    let n_lit = parse_macro_input!(input as LitInt);
    let num_sizes: usize = match n_lit.base10_parse() {
        Ok(n) => n,
        Err(e) => return syn::Error::new(n_lit.span(), e).to_compile_error().into(),
    };

    if num_sizes < 2 {
        return syn::Error::new(
            n_lit.span(),
            "At least 2 sizes required (input and output); this generates num_sizes-1 layers",
        )
        .to_compile_error()
        .into();
    }

    let span = proc_macro2::Span::call_site();
    let gens: Vec<syn::Ident> = (1..=num_sizes).map(|i| syn::Ident::new(&format!("N{}", i), span)).collect();
    let gens_with_const: Vec<_> = gens.iter().map(|g| quote! { const #g: usize }).collect();

    let num_layers = num_sizes - 1;
    let layer_names: Vec<syn::Ident> = (1..=num_layers).map(|i| syn::Ident::new(&format!("l{}", i), span)).collect();

    let layer_fields = (0..num_layers).map(|i| {
        let li = &layer_names[i];
        let inn = &gens[i];
        let out = &gens[i + 1];
        quote! { #li: Layer::<#inn, #out> }
    });

    let act_params: Vec<syn::Ident> = (1..=num_layers).map(|i| syn::Ident::new(&format!("act{}", i), span)).collect();
    let act_params_with_type = act_params.iter().map(|a| quote! { #a: Activations });

    let layer_inits = (0..num_layers).map(|i| {
        let li = &layer_names[i];
        let inn = &gens[i];
        let out = &gens[i + 1];
        let act = &act_params[i];
        quote! { #li: Layer::<#inn, #out>::new(#act) }
    });

    let forward_expr = layer_names.iter().fold(quote! { x }, |acc, layer| {
        quote! { self.#layer.forward(& #acc) }
    });

    let params_expr = {
        let first_layer = &layer_names[0];
        layer_names[1..].iter().fold(quote! { self.#first_layer.parameters() }, |acc, layer| {
            quote! { #acc.chain(self.#layer.parameters()) }
        })
    };

    let last_gen = &gens[num_sizes - 1];

    quote! {
        pub struct MLP< #( #gens_with_const ),* > {
            #( #layer_fields ),*
        }

        impl< #( #gens_with_const ),* > MLP< #( #gens ),* > {
            pub fn new( #( #act_params_with_type ),* ) -> Self {
                Self { #( #layer_inits ),* }
            }

            pub fn forward(&self, x: &[Value; N1]) -> [Value; #last_gen] {
                #forward_expr
            }

            pub fn parameters(&self) -> impl Iterator<Item = &Value> {
                #params_expr
            }
        }
    }
    .into()
}
